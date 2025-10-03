"""Convert MP3 files to WAV format for transcription."""

import sys
from pathlib import Path
from pydub import AudioSegment
import static_ffmpeg

def convert_mp3_to_wav(mp3_path: str, output_path: str = None):
    """Convert MP3 to WAV format.

    Args:
        mp3_path: Path to MP3 file
        output_path: Optional output WAV path (defaults to same name with .wav extension)
    """
    mp3_file = Path(mp3_path)

    if not mp3_file.exists():
        print(f"❌ Error: File not found: {mp3_path}")
        return False

    if output_path is None:
        output_path = mp3_file.with_suffix('.wav')
    else:
        output_path = Path(output_path)

    try:
        # Add static_ffmpeg binaries to PATH
        print("🔧 Setting up ffmpeg...")
        static_ffmpeg.add_paths()

        print(f"📂 Loading: {mp3_file.name}")
        audio = AudioSegment.from_mp3(str(mp3_file))

        print(f"🔄 Converting to WAV (16kHz, mono)")
        # Convert to 16kHz mono for optimal transcription
        audio = audio.set_frame_rate(16000).set_channels(1)

        print(f"💾 Saving: {output_path}")
        audio.export(str(output_path), format='wav')

        print(f"✅ Conversion complete: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print(f"\n💡 The MP3 file might be corrupted. Try playing it in a media player first.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_mp3_to_wav.py <mp3_file> [output_wav_file]")
        print("\nExample:")
        print("  python convert_mp3_to_wav.py audio/sample.mp3")
        print("  python convert_mp3_to_wav.py audio/sample.mp3 audio/output.wav")
        sys.exit(1)

    mp3_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_mp3_to_wav(mp3_path, output_path)
    sys.exit(0 if success else 1)
