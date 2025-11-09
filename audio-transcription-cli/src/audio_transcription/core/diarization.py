"""WhisperX diarization implementation with real speaker detection."""

import whisperx
import torch
import gc
import logging
import librosa
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


def load_audio_with_static_ffmpeg(file_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio using static-ffmpeg if available, otherwise fall back to librosa.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 16000)

    Returns:
        Audio as numpy array (float32)
    """
    try:
        # Try to use static-ffmpeg from venv
        import static_ffmpeg
        ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()

        logger.info(f"Using static-ffmpeg from: {ffmpeg_path}")

        # Use FFmpeg to load and convert audio (same as WhisperX)
        cmd = [
            ffmpeg_path,
            "-nostdin",
            "-threads", "0",
            "-i", file_path,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]

        result = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        logger.info(f"Audio loaded with static-ffmpeg: {len(audio)} samples at {sr}Hz")
        return audio

    except (ImportError, Exception) as e:
        # Fall back to librosa if static-ffmpeg fails
        logger.info(f"static-ffmpeg not available or failed ({e}), using librosa instead")
        audio, sample_rate = librosa.load(file_path, sr=sr)
        logger.info(f"Audio loaded with librosa: {len(audio)} samples at {sample_rate}Hz")
        return audio


def smooth_speaker_labels(segments: List[Dict], window_size: int = 3) -> List[Dict]:
    """Apply majority voting to reduce speaker label flickering.

    This function reduces errors where the same speaker gets different IDs
    in consecutive segments by using a sliding window majority vote.

    Args:
        segments: List of segment dictionaries with speaker labels
        window_size: Size of the sliding window (default: 3)

    Returns:
        Segments with smoothed speaker labels

    Example:
        Before: [SPEAKER_0, SPEAKER_1, SPEAKER_0, SPEAKER_0]
        After:  [SPEAKER_0, SPEAKER_0, SPEAKER_0, SPEAKER_0]
    """
    if len(segments) < window_size or window_size < 2:
        return segments

    smoothed_segments = segments.copy()

    for i in range(len(smoothed_segments)):
        # Only smooth segments that have a speaker label
        if "speaker" not in smoothed_segments[i]:
            continue

        # Define window boundaries
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(smoothed_segments), i + window_size // 2 + 1)

        # Collect speaker labels from window
        window_speakers = [
            seg.get("speaker")
            for seg in smoothed_segments[start_idx:end_idx]
            if "speaker" in seg
        ]

        # Assign most common speaker in window
        if window_speakers:
            most_common_speaker = Counter(window_speakers).most_common(1)[0][0]
            smoothed_segments[i]["speaker"] = most_common_speaker

    logger.debug(f"Applied speaker label smoothing with window size {window_size}")
    return smoothed_segments


def merge_short_segments(segments: List[Dict], min_duration: float = 0.5) -> List[Dict]:
    """Merge segments shorter than min_duration with adjacent segments.

    Pyannote sometimes creates very short segments (0.2-0.5s) when speakers
    briefly overlap. These are often errors and should be merged with
    surrounding segments from the same speaker.

    Args:
        segments: List of segment dictionaries
        min_duration: Minimum segment duration in seconds

    Returns:
        List of segments with short ones merged

    Example:
        Before: [2.5s SPEAKER_0, 0.3s SPEAKER_1, 3.0s SPEAKER_0]
        After:  [5.8s SPEAKER_0]  # Middle segment was likely an error
    """
    if not segments:
        return segments

    merged = []
    i = 0

    while i < len(segments):
        current = segments[i].copy()
        duration = current.get("end", 0.0) - current.get("start", 0.0)

        # If segment is too short and we have a previous segment
        if duration < min_duration and merged:
            prev_speaker = merged[-1].get("speaker", "")
            current_speaker = current.get("speaker", "")

            # Merge with previous if same speaker
            if prev_speaker == current_speaker:
                merged[-1]["end"] = current["end"]
                merged[-1]["text"] = (merged[-1].get("text", "") + " " + current.get("text", "")).strip()
                # Merge words if present
                if "words" in merged[-1] and "words" in current:
                    merged[-1]["words"].extend(current.get("words", []))
                logger.debug(f"Merged short segment ({duration:.2f}s) with previous segment")
                i += 1
                continue

        merged.append(current)
        i += 1

    segments_merged = len(segments) - len(merged)
    if segments_merged > 0:
        logger.info(f"Merged {segments_merged} short segments")

    return merged


def remove_speaker_overlap(segments: List[Dict]) -> List[Dict]:
    """Remove impossible speaker overlaps in single-channel audio.

    Sometimes pyannote assigns overlapping timestamps to different speakers
    (e.g., SPEAKER_0 from 5.0-7.0s, SPEAKER_1 from 6.5-8.0s). In single-channel
    audio, this is impossible and indicates an error. This function resolves
    overlaps by splitting at the midpoint.

    Args:
        segments: List of segment dictionaries

    Returns:
        List of segments with overlaps resolved
    """
    if len(segments) < 2:
        return segments

    cleaned = [segments[0].copy()]
    overlaps_fixed = 0

    for segment in segments[1:]:
        segment_copy = segment.copy()
        prev = cleaned[-1]

        # Check for overlap
        if segment_copy["start"] < prev["end"]:
            # Resolve overlap: split at midpoint
            midpoint = (prev["end"] + segment_copy["start"]) / 2
            prev["end"] = midpoint
            segment_copy["start"] = midpoint
            overlaps_fixed += 1
            logger.debug(f"Resolved speaker overlap at {midpoint:.2f}s")

        cleaned.append(segment_copy)

    if overlaps_fixed > 0:
        logger.info(f"Resolved {overlaps_fixed} speaker overlaps")

    return cleaned


def validate_and_clean_diarization(segments: List[Dict]) -> List[Dict]:
    """Validate and clean diarization results with comprehensive post-processing.

    This function applies multiple cleaning steps:
    1. Remove impossible speaker overlaps
    2. Merge very short segments (likely errors)
    3. Apply speaker label smoothing
    4. Assign speaker labels to unlabeled segments

    Args:
        segments: List of segment dictionaries from diarization

    Returns:
        Cleaned and validated segments
    """
    if not segments:
        logger.warning("No segments to validate")
        return segments

    logger.info(f"Starting post-processing validation on {len(segments)} segments")

    # Step 1: Check for unlabeled segments and assign from neighbors
    unlabeled = [i for i, seg in enumerate(segments) if "speaker" not in seg or not seg.get("speaker")]
    if unlabeled:
        logger.warning(f"Found {len(unlabeled)} unlabeled segments, assigning from neighbors")
        for i in unlabeled:
            if i > 0 and "speaker" in segments[i-1]:
                segments[i]["speaker"] = segments[i-1]["speaker"]
            elif i < len(segments) - 1 and "speaker" in segments[i+1]:
                segments[i]["speaker"] = segments[i+1]["speaker"]
            else:
                segments[i]["speaker"] = "SPEAKER_00"  # Default fallback

    # Step 2: Remove impossible overlaps
    segments = remove_speaker_overlap(segments)

    # Step 3: Merge very short segments
    segments = merge_short_segments(segments, min_duration=0.5)

    # Step 4: Apply speaker label smoothing
    segments = smooth_speaker_labels(segments, window_size=3)

    # Log final speaker statistics
    speakers = sorted(set(seg.get("speaker") for seg in segments if "speaker" in seg))
    logger.info(f"Post-processing complete: {len(segments)} segments, {len(speakers)} speakers detected: {speakers}")

    return segments


class WhisperXDiarizer:
    """WhisperX-based speaker diarization service with real speaker detection."""

    def __init__(self, device: str = "cuda", hf_token: Optional[str] = None, model_size: str = "large-v3"):
        """Initialize WhisperX diarizer.

        Args:
            device: Processing device (cuda or cpu)
            hf_token: HuggingFace token for speaker diarization models
            model_size: WhisperX model size to use (default: large-v3)
        """
        self.device = device
        self.hf_token = hf_token
        self.model_size = model_size
        self.compute_type = "float16" if device == "cuda" else "float32"

        # Models will be loaded on first use (lazy loading)
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

        if not hf_token:
            raise ValueError("HuggingFace token required for speaker diarization")

        logger.info(f"WhisperXDiarizer initialized - Device: {device}, Model: {model_size}")

    def diarize(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform transcription with real WhisperX speaker diarization.

        This implements the full WhisperX pipeline:
        1. Transcribe with WhisperX (batched inference)
        2. Align whisper output for word-level timestamps
        3. Perform speaker diarization with pyannote
        4. Assign speaker labels to words/segments

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es'). Auto-detected if None
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)

        Returns:
            Dictionary with diarization results including segments with speaker labels
        """
        try:
            logger.info(f"Starting WhisperX diarization pipeline for: {audio_path}")

            # Step 1: Load WhisperX model
            logger.info(f"Step 1/5: Loading WhisperX model ({self.model_size})")
            if self.model is None:
                self.model = whisperx.load_model(
                    self.model_size,
                    self.device,
                    compute_type=self.compute_type
                )
                logger.info("WhisperX model loaded successfully")

            # Step 2: Load and transcribe audio
            logger.info(f"Step 2/5: Loading audio and transcribing")
            # Use our custom loader that tries static-ffmpeg first, then librosa
            audio = load_audio_with_static_ffmpeg(str(audio_path), sr=16000)

            # Transcribe with batched inference
            batch_size = 16 if self.device == "cuda" else 8
            transcribe_options = {"batch_size": batch_size}
            if language:
                transcribe_options["language"] = language

            result = self.model.transcribe(audio, **transcribe_options)
            logger.info(f"Transcription complete. Language: {result.get('language', 'unknown')}")

            # Get detected language if not provided
            detected_language = language or result.get("language", "en")

            # Clean up transcription model if memory is tight
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Step 3: Align whisper output for word-level timestamps
            logger.info(f"Step 3/5: Aligning transcription for word-level timestamps")
            if self.align_model is None or self.align_metadata is None:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.device
                )
                logger.info(f"Alignment model loaded for language: {detected_language}")

            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            logger.info("Alignment complete - word-level timestamps generated")

            # Clean up alignment model if memory is tight
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Step 4: Perform speaker diarization
            logger.info(f"Step 4/5: Performing speaker diarization")
            if self.diarize_model is None:
                # Try to load the diarization model with proper error handling
                # Use speaker-diarization-3.1 which is designed for pyannote.audio 3.x
                # This version removes problematic onnxruntime usage and runs in pure PyTorch
                try:
                    self.diarize_model = whisperx.DiarizationPipeline(
                        model_name="pyannote/speaker-diarization-3.1",  # Native v3.1 model for pyannote.audio 3.x
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                    logger.info("Diarization pipeline loaded successfully (pyannote/speaker-diarization-3.1)")
                except Exception as e:
                    logger.error(f"Failed to load diarization pipeline: {e}")
                    # Try alternative import path with explicit model version
                    try:
                        from whisperx.diarize import DiarizationPipeline
                        self.diarize_model = DiarizationPipeline(
                            model_name="pyannote/speaker-diarization-3.1",  # Native v3.1 model
                            use_auth_token=self.hf_token,
                            device=self.device
                        )
                        logger.info("Diarization pipeline loaded via alternative import (pyannote/speaker-diarization-3.1)")
                    except Exception as e2:
                        logger.error(f"Alternative import also failed: {e2}")
                        raise RuntimeError(
                            "Failed to load diarization pipeline. "
                            "Please ensure pyannote.audio is installed and you have accepted "
                            "the terms at: https://huggingface.co/pyannote/speaker-diarization-3.1 "
                            "and https://huggingface.co/pyannote/segmentation-3.0"
                        )

            # Run diarization with optional speaker count constraints
            diarize_options = {}
            if min_speakers is not None:
                diarize_options["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_options["max_speakers"] = max_speakers

            logger.info(f"Running diarization with options: {diarize_options}")
            diarize_segments = self.diarize_model(audio, **diarize_options)
            logger.info("Speaker diarization complete")

            # Clean up diarization model if memory is tight
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Step 5: Assign speaker labels to words and segments
            logger.info(f"Step 5/5: Assigning speaker labels to transcript")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info("Speaker assignment complete")

            # Extract speaker information
            segments_with_speakers = []

            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")

                segments_with_speakers.append({
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "text": segment.get("text", "").strip(),
                    "speaker": speaker,
                    "words": segment.get("words", [])
                })

            # Step 6: Apply post-processing to improve accuracy
            logger.info("Applying post-processing to clean and validate diarization results")
            segments_with_speakers = validate_and_clean_diarization(segments_with_speakers)

            # Extract final speaker information after cleaning
            speakers_found = set()
            for segment in segments_with_speakers:
                speaker = segment.get("speaker", "UNKNOWN")
                speakers_found.add(speaker)

            # Generate formatted transcript
            formatted_transcript = self._format_transcript(segments_with_speakers)

            logger.info(f"Diarization complete: {len(segments_with_speakers)} segments, {len(speakers_found)} speakers")

            return {
                "success": True,
                "segments": segments_with_speakers,
                "num_speakers": len(speakers_found),
                "speakers": sorted(list(speakers_found)),
                "language": detected_language,
                "formatted_transcript": formatted_transcript,
                "model": f"whisperx-{self.model_size}",
                "diarization_method": "pyannote-audio"
            }

        except Exception as e:
            logger.exception(f"WhisperX diarization failed for {audio_path}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        finally:
            # Always clean up memory after processing
            self._cleanup_memory()

    def _format_transcript(self, segments) -> str:
        """Format segments into readable transcript with speaker labels.

        Args:
            segments: List of segment dictionaries with speaker labels

        Returns:
            Formatted transcript string
        """
        transcript_lines = []

        for segment in segments:
            start_time = self._format_time(segment["start"])
            speaker = segment["speaker"]
            text = segment["text"].strip()

            if text:
                transcript_lines.append(f"[{start_time}] {speaker}: {text}")

        return "\n".join(transcript_lines)

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (MM:SS)
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _cleanup_memory(self):
        """Clean up GPU/CPU memory between operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Memory cleanup performed")

    def cleanup(self):
        """Clean up all loaded models and free memory.

        This should be called when done with the diarizer to free resources.
        """
        models = [
            (self.model, "transcription model"),
            (self.align_model, "alignment model"),
            (self.diarize_model, "diarization model")
        ]

        for model, name in models:
            if model is not None:
                try:
                    del model
                    logger.info(f"Cleaned up {name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {e}")

        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("WhisperXDiarizer fully cleaned up")
