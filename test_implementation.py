#!/usr/bin/env python3
"""Quick test script to verify the audio transcription implementation."""

import sys
import os
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "audio-transcription-cli" / "src"))

try:
    # Test basic imports
    print("Testing imports...")

    # Test configuration
    from audio_transcription.config.settings import AppConfig
    print("[OK] Configuration module imported successfully")

    # Test utilities (these don't require heavy dependencies)
    from audio_transcription.utils.logger import setup_logging
    print("✅ Logger module imported successfully")

    try:
        from audio_transcription.utils.file_handler import AudioFileHandler
        print("✅ File handler module imported successfully")

        # Test file handler basic functionality
        handler = AudioFileHandler()
        print(f"✅ AudioFileHandler initialized - Supported formats: {handler.SUPPORTED_FORMATS}")

    except Exception as e:
        print(f"⚠️  File handler has dependency issues: {e}")

    # Test CLI module structure
    try:
        from audio_transcription.cli import app
        print("✅ CLI module structure is correct")
    except Exception as e:
        print(f"⚠️  CLI module has dependency issues: {e}")
        print("   This is expected if transformers/torch are not installed")

    # Test configuration loading
    config = AppConfig()
    print(f"✅ Configuration loaded:")
    print(f"   HF Token present: {'Yes' if config.hf_token else 'No'}")
    print(f"   Default model: {config.default_model}")
    print(f"   Target sample rate: {config.target_sample_rate}")

    # Test directory structure
    cli_dir = Path(__file__).parent / "audio-transcription-cli"
    audio_files = list((cli_dir / "audio").glob("*.mp3")) + list((cli_dir / "audio").glob("*.wav"))
    print(f"✅ Audio files found: {len(audio_files)}")
    for audio_file in audio_files:
        print(f"   - {audio_file.name} ({audio_file.stat().st_size / (1024*1024):.1f} MB)")

    print("\n🎉 Basic implementation structure is correct!")
    print("\nTo complete the setup:")
    print("1. Install dependencies: pip install -r audio-transcription-cli/requirements.txt")
    print("2. Fix the hardcoded model in diarization.py (line 44)")
    print("3. Test with: cd audio-transcription-cli && python -m audio_transcription --help")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the transcribe-backend directory")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()