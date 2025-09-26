"""File handling utilities for audio processing."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Set, List
import librosa
import soundfile as sf
import mutagen
from mutagen import FileType

logger = logging.getLogger(__name__)


class AudioFileHandler:
    """Handles audio file operations."""
    
    SUPPORTED_FORMATS: Set[str] = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
    TARGET_SAMPLE_RATE: int = 16000
    
    def validate_audio_file(self, file_path: Path) -> Dict[str, Any]:
        """Comprehensive audio file validation.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': False,
            'format': None,
            'duration': 0,
            'sample_rate': 0,
            'channels': 0,
            'errors': []
        }
        
        try:
            # Check file existence
            if not file_path.exists():
                validation_results['errors'].append('File does not exist')
                return validation_results
            
            # Check file extension
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                supported = ', '.join(self.SUPPORTED_FORMATS)
                validation_results['errors'].append(
                    f'Unsupported format: {file_path.suffix}. Supported: {supported}'
                )
                return validation_results
            
            # Check file size (max 500MB)
            file_size = file_path.stat().st_size
            if file_size > 500 * 1024 * 1024:
                validation_results['errors'].append('File too large (>500MB)')
                return validation_results
            
            # Validate audio content with mutagen
            try:
                audio_file = mutagen.File(file_path)
                if audio_file is None:
                    validation_results['errors'].append('Not a valid audio file')
                    return validation_results
                
                # Get basic info
                validation_results['duration'] = getattr(audio_file, 'length', 0)
                validation_results['format'] = file_path.suffix.lower()
                
            except Exception as e:
                validation_results['errors'].append(f'Audio validation error: {str(e)}')
                return validation_results
            
            # Try loading with librosa for deeper validation
            try:
                y, sr = librosa.load(str(file_path), duration=1.0)  # Load only 1 second
                if len(y) == 0:
                    validation_results['errors'].append('Empty audio data')
                    return validation_results
                
                validation_results['sample_rate'] = sr
                validation_results['channels'] = 1 if y.ndim == 1 else y.shape[0]
                
            except Exception as e:
                validation_results['errors'].append(f'Librosa loading error: {str(e)}')
                return validation_results
            
            # Check duration limits
            if validation_results['duration'] < 0.1:
                validation_results['errors'].append('Audio too short (<0.1s)')
                return validation_results
            
            if validation_results['duration'] > 3600:  # 1 hour
                validation_results['errors'].append('Audio too long (>1 hour)')
                return validation_results
            
            validation_results['is_valid'] = True
            
        except Exception as e:
            logger.exception(f"Validation error for {file_path}")
            validation_results['errors'].append(f'Validation exception: {str(e)}')
        
        return validation_results
    
    def load_audio(self, file_path: Path) -> Dict[str, Any]:
        """Load and preprocess audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio loading results
        """
        try:
            # Load audio with librosa (handles multiple formats)
            y, sr = librosa.load(
                str(file_path),
                sr=self.TARGET_SAMPLE_RATE,
                mono=True
            )
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=20)
            
            duration = len(y) / self.TARGET_SAMPLE_RATE
            
            return {
                "success": True,
                "audio_data": y,
                "sample_rate": self.TARGET_SAMPLE_RATE,
                "duration": duration,
                "shape": y.shape
            }
            
        except Exception as e:
            logger.exception(f"Failed to load audio: {file_path}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_transcription_result(self, result: Dict[str, Any], output_file: Path):
        """Save transcription result to JSON file."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcription saved to {output_file}")
            
        except Exception as e:
            logger.exception(f"Failed to save transcription: {output_file}")
            raise
    
    def save_diarization_result(self, result: Dict[str, Any], output_file: Path):
        """Save diarization result to JSON file."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Diarization saved to {output_file}")
            
        except Exception as e:
            logger.exception(f"Failed to save diarization: {output_file}")
            raise
    
    def save_text_file(self, text: str, output_file: Path):
        """Save plain text to file."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Text saved to {output_file}")
            
        except Exception as e:
            logger.exception(f"Failed to save text: {output_file}")
            raise
    
    def save_diarized_transcript(self, result: Dict[str, Any], output_file: Path):
        """Save formatted diarized transcript."""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            transcript = result.get("formatted_transcript", "")
            if not transcript and "segments" in result:
                # Generate transcript if not provided
                transcript = self._generate_transcript(result["segments"])
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Add header
                f.write(f"Speaker Diarization Transcript\n")
                f.write(f"Model: {result.get('model', 'unknown')}\n")
                f.write(f"Language: {result.get('language', 'unknown')}\n")
                f.write(f"Speakers detected: {result.get('num_speakers', 'unknown')}\n")
                f.write(f"Speakers: {', '.join(result.get('speakers', []))}\n")
                f.write("="*50 + "\n\n")
                
                # Add transcript
                f.write(transcript)
            
            logger.info(f"Diarized transcript saved to {output_file}")
            
        except Exception as e:
            logger.exception(f"Failed to save diarized transcript: {output_file}")
            raise
    
    def _generate_transcript(self, segments: List[Dict[str, Any]]) -> str:
        """Generate formatted transcript from segments."""
        transcript_lines = []
        
        for segment in segments:
            start_time = self._format_time(segment.get("start", 0))
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            
            if text:
                transcript_lines.append(f"[{start_time}] {speaker}: {text}")
        
        return "\n".join(transcript_lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"