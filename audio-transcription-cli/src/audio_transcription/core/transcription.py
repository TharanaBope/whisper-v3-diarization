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
            
            # Load model with optimizations
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
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
            
            # Prepare generation kwargs
            generate_kwargs = {
                "max_new_tokens": 448,
                "num_beams": 1,
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.35,
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "return_timestamps": True,
            }
            
            if language:
                generate_kwargs["language"] = language
            
            # Transcribe
            logger.info(f"Transcribing {audio_path}")
            result = self.pipe(
                str(audio_path),
                generate_kwargs=generate_kwargs,
                return_timestamps="word"
            )
            
            return {
                "success": True,
                "text": result["text"],
                "chunks": result.get("chunks", []),
                "language": language or "auto-detected",
                "model": self.model_size,
                "timestamp": str(torch.datetime.datetime.now())
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