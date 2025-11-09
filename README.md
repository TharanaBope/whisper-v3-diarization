# Whisper V3 Audio Transcription & Diarization

A powerful, production-ready audio transcription and speaker diarization system with **both CLI and GUI interfaces**. Built with OpenAI Whisper large-v3 and WhisperX.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

## Overview

This project provides **two production-ready interfaces** for audio transcription and speaker diarization:

- **🖥️ Command-Line Interface (CLI)**: For power users, automation, and batch processing
- **🌐 Web GUI**: User-friendly interface with real-time progress tracking and visual results

Both interfaces use the same robust backend processing engine, ensuring consistent quality and features.

## ✨ Features

### Core Processing
- **Advanced Audio Transcription**: High-accuracy transcription using OpenAI Whisper (large-v3)
- **Speaker Diarization**: Identify and label different speakers using WhisperX with pyannote-audio 3.x
- **Enhanced Preprocessing**: Noise reduction, normalization, and spectral subtraction
- **Intelligent Chunking**: 30-second overlapping chunks with smart merging to prevent repetition
- **Multiple Output Formats**: JSON and plain text transcripts with timestamps
- **Comprehensive Audio Support**: MP3, WAV, FLAC, OGG, M4A, AAC (up to 500MB)
- **GPU Acceleration**: CUDA support for 10-50x faster processing
- **Distil-Whisper Support**: Optional 2-5x speed boost with assistant model

### GUI-Specific Features
- **🎨 Modern Interface**: Clean black/white design optimized for readability
- **📊 Real-time Progress**: Live progress tracking via Server-Sent Events (SSE)
- **📁 Drag-and-Drop Upload**: Easy file selection and upload
- **👁️ Visual Results**: Interactive display of transcription and speaker segments
- **⬇️ Easy Downloads**: One-click download of all result files
- **⚙️ Configurable Options**: Adjust model size, language, speaker count, and more

### CLI-Specific Features
- **⚡ Batch Processing**: Process multiple files in one command
- **🔧 Automation-Ready**: Script-friendly with detailed logging
- **🎯 Granular Control**: Fine-tune every processing parameter
- **📝 Rich Terminal Output**: Beautiful progress indicators and formatted results

## 📋 Table of Contents

- [Choose Your Interface](#choose-your-interface)
- [Quick Start](#quick-start)
  - [CLI Quick Start](#cli-quick-start)
  - [GUI Quick Start](#gui-quick-start)
- [CLI Installation & Usage](#cli-installation--usage)
- [GUI Installation & Usage](#gui-installation--usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Choose Your Interface

| Feature | CLI | GUI |
|---------|-----|-----|
| **Best For** | Power users, automation, batch processing | Beginners, visual learners, one-time processing |
| **Installation** | Python only | Python + Node.js |
| **Setup Time** | ~5 minutes | ~10 minutes |
| **Usage** | Command-line commands | Web browser interface |
| **Progress Tracking** | Terminal output | Real-time visual progress bar |
| **Batch Processing** | ✅ Native support | ❌ Single file at a time |
| **Automation** | ✅ Script-friendly | ❌ Manual upload |
| **Visual Results** | Text-based | ✅ Interactive web interface |
| **Status** | ✅ Production-ready | ✅ Production-ready |

**Recommendation**:
- Choose **CLI** if you need to process many files, automate workflows, or prefer terminal interfaces
- Choose **GUI** if you want a simple point-and-click experience with visual feedback

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
git clone https://github.com/TharanaBope/whisper-v3-diarization.git
cd whisper-v3-diarization/audio-transcription-cli
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

## ⚡ Quick Start

### CLI Quick Start

After installation (see [CLI Installation](#cli-installation--usage)):

```bash
# Basic transcription
audio-transcription transcribe path/to/audio.mp3 --model large-v3 --language en

# Speaker diarization
audio-transcription diarize path/to/audio.mp3 --min-speakers 2 --max-speakers 4 --language en

# Full pipeline (transcription + diarization)
audio-transcription process path/to/audio.mp3 --model large-v3 --language en
```

### GUI Quick Start

After installation (see [GUI Installation](#gui-installation--usage)):

1. **Start the backend server**:
   ```bash
   cd gui-app/backend
   # Activate CLI venv
   ../../audio-transcription-cli/venv/Scripts/Activate.ps1  # Windows
   # source ../../audio-transcription-cli/venv/bin/activate  # macOS/Linux
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the frontend** (in a new terminal):
   ```bash
   cd gui-app/frontend
   npm run dev
   ```

3. **Open your browser** to `http://localhost:3000`

4. **Upload and process**:
   - Drag and drop your audio file or click to browse
   - Configure options (model, language, speaker count)
   - Click "Start Processing" and watch real-time progress
   - View and download results when complete

---

## 🖥️ CLI Installation & Usage

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

---

## 🌐 GUI Installation & Usage

The GUI provides a modern web interface for audio transcription and diarization with real-time progress tracking.

### Prerequisites

- **All CLI prerequisites** (Python 3.9+, FFmpeg, optional CUDA, HuggingFace token)
- **Node.js 18+** and **npm 9+**: [Download from nodejs.org](https://nodejs.org/)

### Installation

1. **Complete CLI installation first** (see above) - the GUI shares the CLI's processing modules

2. **Install GUI backend dependencies**:
   ```bash
   cd gui-app/backend
   # Activate CLI's virtual environment
   ../../audio-transcription-cli/venv/Scripts/Activate.ps1  # Windows
   # source ../../audio-transcription-cli/venv/bin/activate  # macOS/Linux

   pip install -r requirements.txt
   ```

3. **Configure GUI backend**:
   ```bash
   # Create .env file in gui-app/backend/
   cat > .env << EOF
   HF_TOKEN=your_huggingface_token_here
   CUDA_VISIBLE_DEVICES=0
   API_HOST=0.0.0.0
   API_PORT=8000
   EOF
   ```

4. **Install frontend dependencies**:
   ```bash
   cd ../frontend
   npm install
   ```

5. **Configure frontend**:
   ```bash
   echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
   ```

### Running the GUI

**Terminal 1 - Backend**:
```bash
cd gui-app/backend
../../audio-transcription-cli/venv/Scripts/Activate.ps1  # Windows
# source ../../audio-transcription-cli/venv/bin/activate  # macOS/Linux
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend**:
```bash
cd gui-app/frontend
npm run dev
```

**Access**: Open `http://localhost:3000` in your browser

### GUI Features

- **📁 File Upload**: Drag-and-drop or click to browse for audio files
- **⚙️ Configuration Options**:
  - Whisper model size (tiny to large-v3)
  - Language selection (auto-detect or specify)
  - Speaker diarization enable/disable
  - Min/max speaker count
  - Device selection (auto/CPU/GPU)
  - Distil-Whisper assistant toggle
- **📊 Real-time Progress**: Live progress bar with stage-by-stage updates
- **👁️ Visual Results**:
  - Speaker statistics
  - Formatted transcription with speaker labels
  - Timestamp information
- **⬇️ Download Options**:
  - Transcription (TXT, JSON)
  - Diarization (TXT, JSON)

### GUI Architecture

```
┌─────────────────────────────────────┐
│   Next.js 15 Frontend (Port 3000)  │
│   - React 19 + TypeScript           │
│   - Tailwind CSS (black/white)     │
│   - SSE for real-time updates      │
└──────────────┬──────────────────────┘
               │ HTTP/SSE
┌──────────────┴──────────────────────┐
│   FastAPI Backend (Port 8000)      │
│   - File upload & session mgmt     │
│   - Progress streaming (SSE)       │
│   - Results retrieval              │
└──────────────┬──────────────────────┘
               │ Python imports
┌──────────────┴──────────────────────┐
│   Shared CLI Core Modules          │
│   - AudioProcessor                 │
│   - WhisperX Diarization           │
│   - Transcription Engine           │
└─────────────────────────────────────┘
```

**For detailed GUI setup and troubleshooting**: See [`gui-app/QUICKSTART.md`](gui-app/QUICKSTART.md)

---

## ⚙️ Configuration

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

## 🏗️ Architecture

### Project Structure

```
whisper-v3-diarization/
├── README.md                          # Main documentation
├── SETUP_GUIDE.md                    # Installation guide
├── LICENSE                           # MIT License
├── .gitignore                       # Git ignore rules
│
├── audio-transcription-cli/          # ✅ CLI Application
│   ├── src/audio_transcription/
│   │   ├── core/                    # ← Shared processing modules
│   │   │   ├── audio_processor.py  # Main coordinator
│   │   │   ├── transcription.py    # Whisper transcription
│   │   │   └── diarization.py      # WhisperX diarization
│   │   ├── utils/
│   │   │   ├── file_handler.py     # Audio I/O
│   │   │   └── logger.py           # Logging
│   │   ├── config/settings.py      # Configuration
│   │   └── cli.py                  # CLI interface
│   ├── audio/                       # Sample audio files
│   ├── transcriptions/              # Output directory
│   └── diarizations/                # Output directory
│
└── gui-app/                          # ✅ GUI Application
    ├── backend/                     # FastAPI server
    │   ├── main.py                 # App entry point
    │   ├── config.py               # Settings
    │   ├── models.py               # Pydantic models
    │   ├── routes/                 # API endpoints
    │   │   ├── upload.py           # File upload
    │   │   ├── process.py          # SSE progress
    │   │   └── results.py          # Results retrieval
    │   ├── services/
    │   │   └── processor.py        # Wraps core modules
    │   └── storage/                # Uploads & results
    │
    └── frontend/                    # Next.js application
        ├── app/
        │   ├── page.tsx            # Upload page
        │   └── results/[id]/       # Results viewer
        ├── components/
        │   ├── FileUpload.tsx      # Drag-drop upload
        │   ├── ProgressStream.tsx  # SSE client
        │   └── ResultsViewer.tsx   # Display results
        └── lib/
            ├── api.ts              # API client
            └── types.ts            # TypeScript types
```

### Key Technologies

**Backend Processing**:
- **OpenAI Whisper** - State-of-the-art speech recognition (large-v3)
- **WhisperX** - Enhanced Whisper with word-level timestamps and diarization
- **pyannote.audio 3.x** - Speaker diarization models
- **Librosa** - Audio processing and analysis
- **PyTorch + CUDA** - Deep learning framework with GPU support

**CLI Interface**:
- **Typer** - Modern CLI framework
- **Rich** - Beautiful terminal output

**GUI Stack**:
- **FastAPI** - Modern Python web framework for backend API
- **Next.js 15** - React 19 framework with App Router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS v3.4.14** - Utility-first CSS (black/white theme)
- **Server-Sent Events (SSE)** - Real-time progress streaming

### Processing Pipeline

1. **Audio Validation** - Check format, duration, and integrity
2. **Preprocessing** - Noise reduction, normalization, resampling to 16kHz
3. **Chunking** - Split audio into 30-second overlapping segments
4. **Transcription** - Process each chunk with Whisper
5. **Merging** - Intelligently combine chunks, removing overlaps
6. **Diarization** (optional) - Identify speakers using WhisperX
7. **Output Generation** - Save JSON and text formats

## 🔧 Troubleshooting

### CLI Issues

**Issue: `ModuleNotFoundError: No module named 'audio_transcription'`**
```bash
# Solution: Install package in editable mode
cd audio-transcription-cli
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
audio-transcription transcribe audio.wav --model medium
```

**Issue: Diarization hanging or taking too long**
```bash
# Solution: The system has automatic fallbacks
# If WhisperX VAD hangs, it uses sentence-based speaker assignment
# Just wait, or interrupt and check logs
```

**Issue: MP3 files not loading correctly**
```bash
# Solution: Convert to WAV first
cd audio-transcription-cli
python convert_mp3_to_wav.py audio/problematic.mp3
```

### GUI Issues

**Issue: Backend import errors (`ImportError: attempted relative import`)**
```bash
# Solution: Ensure all backend imports are absolute, not relative
# Check: main.py, routes/*.py, services/*.py
# Should use: from config import settings
# NOT: from .config import settings
```

**Issue: `Pydantic validation error` for extra fields**
```python
# Solution: Add fields to config.py Settings class and use extra="ignore"
class Settings(BaseSettings):
    CUDA_VISIBLE_DEVICES: str = "0"

    class Config:
        env_file = ".env"
        extra = "ignore"
```

**Issue: Cannot access `http://0.0.0.0:8000`**
```
Solution: Use http://localhost:8000 instead
(0.0.0.0 is a bind address for servers, not accessible in browsers)
```

**Issue: TypeScript CSS import error**
```bash
# Solution: Create global.d.ts in frontend directory
# See gui-app/QUICKSTART.md for details
```

**Issue: UI text not visible (too light)**
```bash
# Solution: Clear browser cache and hard refresh (Ctrl+F5)
# The black/white theme should make all text clearly visible
```

**Issue: `KeyError: 'filename'` in backend logs**
```python
# Solution: Ensure session dict includes filename in services/processor.py
self.active_sessions[session_id] = {
    "status": "processing",
    "audio_path": audio_path,
    "filename": audio_path.name  # This line is required
}
```

**Issue: SSE connection fails / progress not updating**
```bash
# Check backend is running: curl http://localhost:8000/health
# Check browser console for errors
# Ensure no proxy buffering SSE events
```

**For more troubleshooting**: See `gui-app/QUICKSTART.md` and `.claude/deployment.md`

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features for both CLI and GUI.

### Development Setup

**CLI Development**:
```bash
# Clone repository
git clone https://github.com/TharanaBope/whisper-v3-diarization.git
cd whisper-v3-diarization/audio-transcription-cli

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
isort src/
```

**GUI Development**:
```bash
# Backend development
cd gui-app/backend
# Use CLI venv and install GUI dependencies
pip install -r requirements.txt

# Frontend development
cd ../frontend
npm install
npm run dev
```

### Guidelines

- Follow PEP 8 style guidelines for Python code
- Use TypeScript strict mode for frontend code
- Add tests for new features
- Update documentation for API changes
- Use meaningful commit messages
- For GUI changes, test in multiple browsers
- Ensure both CLI and GUI work after core module changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OpenAI** for the Whisper model
- **Max Bain** for WhisperX
- **Pyannote.audio** team for speaker diarization models
- **HuggingFace** for model hosting and transformers library

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{whisper_v3_diarization,
  title = {Whisper V3 Audio Transcription \& Diarization - CLI + GUI},
  author = {TharanaBope},
  year = {2025},
  url = {https://github.com/TharanaBope/whisper-v3-diarization}
}
```

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/TharanaBope/whisper-v3-diarization/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/TharanaBope/whisper-v3-diarization/discussions)
- **Documentation**:
  - CLI Setup: [`SETUP_GUIDE.md`](SETUP_GUIDE.md)
  - GUI Setup: [`gui-app/QUICKSTART.md`](gui-app/QUICKSTART.md)
  - Implementation Details: [`.claude/`](.claude/) directory

---

**Made with ❤️ for the open-source community**
