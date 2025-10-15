# Audio Transcription & Diarization CLI

A powerful, production-ready command-line tool for audio transcription and speaker diarization using OpenAI Whisper and WhisperX.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Advanced Audio Transcription**: High-accuracy transcription using OpenAI Whisper (large-v3)
- **Speaker Diarization**: Identify and label different speakers using WhisperX with pyannote-audio
- **Enhanced Preprocessing**: Noise reduction, normalization, and spectral subtraction
- **Intelligent Chunking**: 30-second overlapping chunks with smart merging to prevent repetition
- **Multiple Output Formats**: JSON and plain text transcripts with timestamps
- **Comprehensive Audio Support**: MP3, WAV, FLAC, OGG, M4A, AAC
- **GPU Acceleration**: CUDA support for faster processing
- **Production-Ready**: Robust error handling, fallback mechanisms, and detailed logging

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.9 or higher
- FFmpeg (for audio processing)
- CUDA-compatible GPU (optional, for faster processing)

### Step 1: Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/TharanaBope/transcribe-backend.git
cd transcribe-backend/audio-transcription-cli
```

### Step 3: Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 5: Configure Environment Variables

Copy the example environment file and add your HuggingFace token:

```bash
cp .env.example .env
```

Edit `.env` and add your HuggingFace token (required for diarization):
```env
HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
LOG_LEVEL=INFO
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

**Important**: You also need to accept the pyannote.audio speaker diarization model terms at:
https://huggingface.co/pyannote/speaker-diarization-3.1

## Quick Start

### Basic Transcription

```bash
audio-transcription transcribe path/to/audio.mp3 --model large-v3 --language en
```

### Speaker Diarization

```bash
audio-transcription diarize path/to/audio.mp3 --min-speakers 2 --max-speakers 4 --language en
```

### Full Pipeline (Transcription + Diarization)

```bash
audio-transcription process path/to/audio.mp3 --model large-v3 --language en
```

## Usage

The CLI provides three main commands:

### 1. `transcribe` - Audio Transcription Only

Transcribe audio files using Whisper with advanced preprocessing.

```bash
audio-transcription transcribe <audio_file> [OPTIONS]
```

**Options:**
- `--model TEXT` - Whisper model size (default: `large-v3`)
  - Options: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`
- `--language TEXT` - Audio language (default: auto-detect)
  - Examples: `en`, `es`, `fr`, `de`, `zh`, `ja`
- `--device TEXT` - Device to use (default: `auto`)
  - Options: `auto`, `cuda`, `cpu`
- `--output-dir PATH` - Output directory (default: `transcriptions/`)

**Example:**
```bash
audio-transcription transcribe audio/interview.mp3 \
  --model large-v3 \
  --language en \
  --output-dir ./results
```

### 2. `diarize` - Speaker Diarization

Identify and label different speakers in audio files.

```bash
audio-transcription diarize <audio_file> [OPTIONS]
```

**Options:**
- `--model TEXT` - Whisper model size (default: `large-v3`)
- `--min-speakers INTEGER` - Minimum number of speakers (default: `1`)
- `--max-speakers INTEGER` - Maximum number of speakers (default: `10`)
- `--language TEXT` - Audio language (default: auto-detect)
- `--device TEXT` - Device to use (default: `auto`)
- `--output-dir PATH` - Output directory (default: `diarizations/`)

**Example:**
```bash
audio-transcription diarize audio/meeting.mp3 \
  --model large-v3 \
  --min-speakers 2 \
  --max-speakers 5 \
  --language en
```

### 3. `process` - Full Pipeline

Run both transcription and diarization in one command.

```bash
audio-transcription process <audio_file> [OPTIONS]
```

**Options:** Combines all options from `transcribe` and `diarize` commands.

**Example:**
```bash
audio-transcription process audio/podcast.mp3 \
  --model large-v3 \
  --min-speakers 2 \
  --max-speakers 3 \
  --language en
```

## Configuration

### Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```env
# HuggingFace token (required for diarization)
HF_TOKEN=your_token_here

# CUDA device selection (0 for first GPU, -1 for CPU)
CUDA_VISIBLE_DEVICES=0

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Suppress warnings (optional)
PYTHONWARNINGS=ignore::UserWarning
```

### Audio Format Support

Supported formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`

Maximum file size: 500MB (configurable in `src/audio_transcription/utils/file_handler.py`)

## Examples

### Convert MP3 to WAV

If you encounter MP3 format issues:

```bash
python convert_mp3_to_wav.py audio/sample.mp3
```

Or use FFmpeg directly:

```bash
ffmpeg -i audio/sample.mp3 -ar 16000 audio/sample.wav
```

### Debug Audio Files

Test audio file validity:

```bash
python debug_audio.py
```

### Batch Processing

Process multiple audio files:

```bash
for file in audio/*.wav; do
    audio-transcription diarize "$file" --min-speakers 2 --max-speakers 4
done
```

### Custom Output Directory

```bash
audio-transcription transcribe audio/sample.wav \
  --output-dir ./my_transcriptions \
  --model large-v3
```

## Architecture

### Project Structure

```
transcribe-backend/
├── README.md                          # Main documentation
├── LICENSE                            # MIT License
├── .gitignore                        # Git ignore rules
├── DIARIZATION_IMPLEMENTATION.md     # Technical docs
└── audio-transcription-cli/          # CLI application
    ├── .env.example                 # Environment template
    ├── pyproject.toml              # Package configuration
    ├── requirements.txt            # Dependencies
    ├── convert_mp3_to_wav.py       # MP3 conversion utility
    ├── debug_audio.py              # Audio debugging tool
    ├── src/audio_transcription/
    │   ├── cli.py                 # CLI interface
    │   ├── __main__.py            # Entry point
    │   ├── config/
    │   │   └── settings.py       # Configuration
    │   ├── core/                  # Core processing
    │   │   ├── audio_processor.py
    │   │   ├── transcription.py
    │   │   └── diarization.py
    │   └── utils/                 # Utilities
    │       ├── file_handler.py
    │       └── logger.py
    └── tests/                      # Test suite
```

### Key Technologies

- **OpenAI Whisper** - State-of-the-art speech recognition
- **WhisperX** - Enhanced Whisper with word-level timestamps and diarization
- **pyannote.audio** - Speaker diarization models
- **Librosa** - Audio processing and analysis
- **Typer** - Modern CLI framework
- **Rich** - Beautiful terminal output

### Processing Pipeline

1. **Audio Validation** - Check format, duration, and integrity
2. **Preprocessing** - Noise reduction, normalization, resampling to 16kHz
3. **Chunking** - Split audio into 30-second overlapping segments
4. **Transcription** - Process each chunk with Whisper
5. **Merging** - Intelligently combine chunks, removing overlaps
6. **Diarization** (optional) - Identify speakers using WhisperX
7. **Output Generation** - Save JSON and text formats

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'audio_transcription'`**
```bash
# Solution: Install package in editable mode
pip install -e .
```

**Issue: `Could not load libtorchcodec` or FFmpeg errors**
```bash
# Solution: The project bypasses this - no action needed
# But ensure FFmpeg is installed if converting audio formats
```

**Issue: CUDA out of memory**
```bash
# Solution 1: Use CPU instead
audio-transcription transcribe audio.wav --device cpu

# Solution 2: Use smaller model
audio-transcription transcribe audio.wav --model large-v2
```

**Issue: Diarization hanging or taking too long**
```bash
# Solution: The system has automatic fallbacks
# If WhisperX VAD hangs, it uses sentence-based speaker assignment
# Just wait, or interrupt and check logs/
```

**Issue: MP3 files not loading correctly**
```bash
# Solution: Convert to WAV first
cd audio-transcription-cli
python convert_mp3_to_wav.py audio/problematic.mp3
```

### Debug Mode

Enable detailed logging:

```bash
# Set LOG_LEVEL=DEBUG in .env
# Or run with verbose output
audio-transcription --log-level DEBUG transcribe audio/sample.wav
```

### Getting Help

```bash
# View all commands
audio-transcription --help

# View command-specific help
audio-transcription transcribe --help
audio-transcription diarize --help
audio-transcription process --help
```

## Performance Considerations

- **GPU vs CPU**: GPU (CUDA) is 10-50x faster than CPU for transcription
- **Model Size**: Larger models (`large-v3`) are more accurate but slower
- **Audio Length**: Processing time scales linearly with audio duration
- **Diarization**: Adds significant processing time (2-5x longer than transcription alone)
- **Memory**: Expect 4-8GB GPU memory usage for `large-v3` model

### Recommended Specifications

**Minimum:**
- Python 3.9+
- 8GB RAM
- CPU: Any modern processor

**Recommended:**
- Python 3.10+
- 16GB RAM
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- Storage: 10GB for models and dependencies

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

### Development Setup

```bash
# Clone repository
git clone https://github.com/TharanaBope/transcribe-backend.git
cd transcribe-backend/audio-transcription-cli

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

### Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Use meaningful commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI** for the Whisper model
- **Max Bain** for WhisperX
- **Pyannote.audio** team for speaker diarization models
- **HuggingFace** for model hosting and transformers library

## Citation

If you use this project in your research, please cite:

```bibtex
@software{audio_transcription_cli,
  title = {Audio Transcription & Diarization CLI},
  author = {TharanaBope},
  year = {2025},
  url = {https://github.com/TharanaBope/transcribe-backend}
}
```

## Contact

- GitHub Issues: [Report a bug](https://github.com/TharanaBope/transcribe-backend/issues)

---

**Made with ❤️ for the open-source community**
