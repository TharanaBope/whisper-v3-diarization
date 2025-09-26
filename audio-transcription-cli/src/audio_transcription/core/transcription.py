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
            audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)

            # Chunk audio to handle long files (30 second chunks)
            chunk_duration = 30  # seconds
            chunk_samples = chunk_duration * sample_rate
            audio_length = len(audio_data)

            logger.info(f"Audio duration: {audio_length / sample_rate:.2f}s, processing in {chunk_duration}s chunks")

            all_transcriptions = []

            for i in range(0, audio_length, chunk_samples):
                chunk_end = min(i + chunk_samples, audio_length)
                audio_chunk = audio_data[i:chunk_end]

                if len(audio_chunk) < sample_rate * 0.5:  # Skip chunks shorter than 0.5 seconds
                    continue

                chunk_num = i // chunk_samples + 1
                total_chunks = (audio_length + chunk_samples - 1) // chunk_samples
                logger.info(f"Processing chunk {chunk_num}/{total_chunks}")

                # Generate mel spectrogram for chunk with correct dtype
                features = self.processor(
                    audio_chunk,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).input_features.to(self.device, dtype=self.torch_dtype)

                # Generate transcription for chunk
                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        features,
                        **generate_kwargs
                    )

                # Decode the transcription
                chunk_transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

                if chunk_transcription.strip():  # Only add non-empty transcriptions
                    all_transcriptions.append(chunk_transcription.strip())

            # Combine all transcriptions
            full_transcription = " ".join(all_transcriptions)

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