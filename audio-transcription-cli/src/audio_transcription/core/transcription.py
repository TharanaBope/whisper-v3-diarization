"""Whisper transcription implementation."""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gc

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Whisper-based transcription service."""
    
    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        """Initialize Whisper transcriber.
        
        Args:
            model_size: Whisper model size
            device: Processing device
        """
        self.model_size = model_size
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Model and pipeline will be loaded on first use
        self.model = None
        self.processor = None
        self.pipe = None
        
        logger.info(f"WhisperTranscriber initialized - Model: {model_size}, Device: {device}")
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is not None:
            return
        
        try:
            model_id = f"openai/whisper-{self.model_size}"
            logger.info(f"Loading Whisper model: {model_id}")
            
            # Load model with optimizations and proper fallback
            try:
                # Try with FlashAttention2 first if on CUDA
                if self.device == "cuda":
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_id,
                        torch_dtype=self.torch_dtype,
                        low_cpu_mem_usage=True,
                        use_safetensors=True,
                        attn_implementation="flash_attention_2"
                    )
                else:
                    self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_id,
                        torch_dtype=self.torch_dtype,
                        low_cpu_mem_usage=True,
                        use_safetensors=True,
                        attn_implementation="eager"
                    )
            except ImportError:
                # Fallback to eager attention if FlashAttention2 not available
                logger.warning("FlashAttention2 not available, falling back to eager attention")
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="eager"
                )
            self.model.to(self.device)
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Create pipeline
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                return_timestamps=True
            )
            
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
        
        Returns:
            Transcription result
        """
        try:
            self._load_model()
            
            # Prepare generation kwargs (reduce max_new_tokens to avoid limit issues)
            generate_kwargs = {
                "max_new_tokens": 200,  # Reduced to avoid token limit
                "num_beams": 1,
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.35,
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            
            if language:
                generate_kwargs["language"] = language
            
            # Load audio data with librosa (bypasses all FFmpeg/TorchCodec issues)
            logger.info(f"Loading audio data from {audio_path}")
            import librosa
            import numpy as np
            raw_audio, sample_rate = librosa.load(str(audio_path), sr=16000)

            # Enhanced audio preprocessing for better accuracy
            logger.info("Applying audio preprocessing enhancements")
            audio_data = self._preprocess_audio(raw_audio, sample_rate)

            # Improved chunking with overlap for better accuracy
            chunk_duration = 30  # seconds
            overlap_duration = 5   # seconds overlap between chunks
            chunk_samples = chunk_duration * sample_rate
            overlap_samples = overlap_duration * sample_rate
            audio_length = len(audio_data)

            logger.info(f"Audio duration: {audio_length / sample_rate:.2f}s, processing with {chunk_duration}s chunks and {overlap_duration}s overlap")

            all_transcriptions = []
            chunk_timestamps = []

            # Process chunks with overlap
            i = 0
            chunk_num = 1
            while i < audio_length:
                chunk_end = min(i + chunk_samples, audio_length)
                audio_chunk = audio_data[i:chunk_end]

                if len(audio_chunk) < sample_rate * 0.5:  # Skip chunks shorter than 0.5 seconds
                    break

                total_chunks = max(1, (audio_length - chunk_samples) // (chunk_samples - overlap_samples) + 1)
                logger.info(f"Processing chunk {chunk_num}/{total_chunks} (samples {i}-{chunk_end})")

                # Enhanced preprocessing for this chunk
                processed_chunk = self._enhance_chunk_audio(audio_chunk, sample_rate)

                # Generate mel spectrogram for chunk with correct dtype
                features = self.processor(
                    processed_chunk,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).input_features.to(self.device, dtype=self.torch_dtype)

                # Generate transcription for chunk with improved parameters
                enhanced_kwargs = generate_kwargs.copy()
                if chunk_num == 1:
                    # Special handling for first chunk to improve accuracy
                    enhanced_kwargs["temperature"] = (0.0, 0.1, 0.2, 0.3)  # Lower temperature for first chunk
                    enhanced_kwargs["no_speech_threshold"] = 0.4  # Lower threshold for first chunk

                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        features,
                        **enhanced_kwargs
                    )

                # Decode the transcription
                chunk_transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

                if chunk_transcription.strip():  # Only add non-empty transcriptions
                    # Store transcription with timing info for overlap handling
                    start_time = i / sample_rate
                    end_time = chunk_end / sample_rate
                    all_transcriptions.append({
                        "text": chunk_transcription.strip(),
                        "start": start_time,
                        "end": end_time,
                        "chunk_num": chunk_num
                    })
                    chunk_timestamps.append((start_time, end_time))

                # Move to next chunk with overlap
                if chunk_end >= audio_length:
                    break
                i += chunk_samples - overlap_samples
                chunk_num += 1

            # Combine transcriptions with overlap handling
            full_transcription = self._merge_overlapping_transcriptions(all_transcriptions)

            # Format result to match expected structure
            result = {"text": full_transcription}
            
            return {
                "success": True,
                "text": result["text"],
                "chunks": [],  # Direct model approach doesn't provide chunks
                "language": language or "auto-detected",
                "model": self.model_size,
                "timestamp": str(torch.datetime.datetime.now()) if hasattr(torch, 'datetime') else str(__import__('datetime').datetime.now())
            }
            
        except Exception as e:
            logger.exception(f"Transcription failed for {audio_path}")
            return {
                "success": False,
                "error": str(e),
                "model": self.model_size
            }
    
    def _preprocess_audio(self, audio_data, sample_rate):
        """Enhanced audio preprocessing for better accuracy."""
        import librosa
        import numpy as np

        logger.info("Applying noise reduction and normalization")

        # Apply spectral gating noise reduction
        try:
            # Reduce noise using spectral subtraction approach
            # Compute power spectrum
            D = librosa.stft(audio_data)
            magnitude = np.abs(D)

            # Estimate noise from first 0.5 seconds
            noise_frame_count = int(0.5 * sample_rate / 512)  # 512 is default hop_length
            noise_profile = np.mean(magnitude[:, :noise_frame_count], axis=1, keepdims=True)

            # Apply spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Residual noise factor
            enhanced_magnitude = magnitude - alpha * noise_profile
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)

            # Reconstruct audio
            enhanced_D = enhanced_magnitude * np.exp(1j * np.angle(D))
            audio_data = librosa.istft(enhanced_D)

        except Exception as e:
            logger.warning(f"Noise reduction failed, using original audio: {e}")

        # Apply adaptive normalization
        # Use RMS-based normalization for consistent loudness
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            target_rms = 0.1  # Target RMS level
            audio_data = audio_data * (target_rms / rms)

        # Apply gentle high-pass filter to remove low-frequency noise
        try:
            audio_data = librosa.effects.preemphasis(audio_data)
        except:
            logger.warning("Preemphasis failed, skipping")

        # Ensure audio doesn't clip
        audio_data = np.clip(audio_data, -1.0, 1.0)

        return audio_data

    def _enhance_chunk_audio(self, chunk_audio, sample_rate):
        """Apply per-chunk audio enhancements."""
        import librosa
        import numpy as np

        # Apply gentle compression to even out dynamics
        # Simple soft compression
        threshold = 0.5
        ratio = 4.0
        above_threshold = np.abs(chunk_audio) > threshold

        compressed = chunk_audio.copy()
        compressed[above_threshold] = np.sign(chunk_audio[above_threshold]) * (
            threshold + (np.abs(chunk_audio[above_threshold]) - threshold) / ratio
        )

        # Apply subtle smoothing to reduce artifacts
        # Simple moving average filter
        window_size = min(3, len(compressed))
        if window_size > 1:
            compressed = np.convolve(compressed, np.ones(window_size)/window_size, mode='same')

        return compressed

    def _merge_overlapping_transcriptions(self, transcriptions):
        """Merge overlapping transcriptions intelligently."""
        if not transcriptions:
            return ""

        if len(transcriptions) == 1:
            return transcriptions[0]["text"]

        merged_text = []

        for i, trans in enumerate(transcriptions):
            text = trans["text"].strip()

            if i == 0:
                # First chunk - use entirely
                merged_text.append(text)
            else:
                # For subsequent chunks, try to remove overlapping content
                prev_text = merged_text[-1] if merged_text else ""

                # Simple overlap detection - find common ending/beginning words
                current_words = text.split()
                prev_words = prev_text.split()

                if len(current_words) > 0 and len(prev_words) > 0:
                    # Look for overlap in last few words of previous and first few words of current
                    max_overlap = min(5, len(current_words), len(prev_words))
                    overlap_found = 0

                    for j in range(1, max_overlap + 1):
                        if prev_words[-j:] == current_words[:j]:
                            overlap_found = j

                    if overlap_found > 0:
                        # Remove overlapping words from current chunk
                        text = " ".join(current_words[overlap_found:])
                        logger.info(f"Detected {overlap_found} word overlap, merging chunks")

                if text.strip():  # Only add if there's remaining content
                    merged_text.append(text.strip())

        return " ".join(merged_text)

    def cleanup(self):
        """Clean up model memory."""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if self.pipe:
            del self.pipe
            self.pipe = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("WhisperTranscriber cleaned up")