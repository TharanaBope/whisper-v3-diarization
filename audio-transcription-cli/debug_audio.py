#!/usr/bin/env python3
"""Debug audio file issues."""

import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, "src")

try:
    import librosa
    import mutagen
    from audio_transcription.utils.file_handler import AudioFileHandler

    # Check both audio files
    audio_dir = Path("audio")
    files = [audio_dir / "sample.mp3", audio_dir / "sample_16k.wav"]

    for file_path in files:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        print(f"\n=== Analyzing {file_path.name} ===")
        print(f"Size: {file_path.stat().st_size / (1024*1024):.2f} MB")

        # Test with mutagen
        try:
            audio_file = mutagen.File(file_path)
            if audio_file:
                duration = getattr(audio_file, 'length', 0)
                print(f"Mutagen duration: {duration:.2f}s")
            else:
                print("Mutagen: Could not read file")
        except Exception as e:
            print(f"Mutagen error: {e}")

        # Test with librosa (small sample)
        try:
            y, sr = librosa.load(str(file_path), duration=1.0)
            print(f"Librosa sample: {len(y)} samples at {sr}Hz = {len(y)/sr:.2f}s")
        except Exception as e:
            print(f"Librosa error: {e}")

        # Test with librosa (full file)
        try:
            y_full, sr_full = librosa.load(str(file_path), sr=None)
            duration_full = len(y_full) / sr_full if len(y_full) > 0 else 0
            print(f"Librosa full: {len(y_full)} samples at {sr_full}Hz = {duration_full:.2f}s")
        except Exception as e:
            print(f"Librosa full error: {e}")

        # Test file handler
        try:
            handler = AudioFileHandler()
            result = handler.validate_audio_file(file_path)
            print(f"Validation result: {result}")
        except Exception as e:
            print(f"Handler error: {e}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Run: pip install mutagen")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()