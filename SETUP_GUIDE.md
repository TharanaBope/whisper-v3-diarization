# Whisper V3 Audio Transcription & Diarization - Complete Setup Guide

This guide covers setup for **both CLI and GUI interfaces**.

## 📋 Table of Contents

- [Choose Your Setup](#choose-your-setup)
- [CLI Setup](#cli-setup-command-line-interface)
- [GUI Setup](#gui-setup-web-interface)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Choose Your Setup

| Setup | Time Required | Best For |
|-------|---------------|----------|
| **CLI Only** | ~5-10 minutes | Power users, automation, batch processing |
| **GUI Only** | ~15-20 minutes | Visual interface, one-time processing |
| **Both** | ~20-25 minutes | Maximum flexibility |

**Note**: GUI setup requires CLI to be installed first (they share core modules).

---

# CLI Setup (Command-Line Interface)

## Step 0: Verify CUDA Installation

```bash
# Check NVIDIA driver and CUDA
nvidia-smi

# Check CUDA toolkit version
nvcc --version

# Check cuDNN is in PATH
#if not available -
- https://developer.nvidia.com/cuda-12-8-0-download-archive

where cudnn64_8.dll
```

## Step 1: Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or (Windows CMD)
.\venv\Scripts\activate.bat
```

**Option B: Using Conda (Alternative)**

```bash
# Create conda environment
conda create -n transcribe python=3.11 -y

# Activate it
conda activate transcribe
```

## Step 2: Install PyTorch with CUDA Support

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

**Verify PyTorch Installation:**

```bash
@"
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
"@ | python
```

**Expected Output:**
PyTorch: 2.8.0+cu128
CUDA Available: True
CUDA Version: 12.8
GPU Count: 1

**✅ If CUDA Available is True, you're good! Proceed to Step 3.**

## Step 3: Install Core Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

## Step 4: Install the CLI Application

```bash
# Make sure you're in the directory
# Install in editable mode
pip install -e .
```

## Step 5: Configure Environment Variables

# Copy the example

copy .env.example .env

# Edit .env and add your HuggingFace token

Your `.env` should contain:

```env
HF_TOKEN=hf_YourActualTokenHere
CUDA_VISIBLE_DEVICES=0
LOG_LEVEL=INFO
PYTHONWARNINGS=ignore::UserWarning
```

```bash
#if error as
#⚠️  Warning: No HuggingFace token provided. Speaker diarization will be disabled.
#❌ HuggingFace token required for speaker diarization!

$env:HF_TOKEN = "hf_YourActualTokenHere"
```

**Get HuggingFace Token:**

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "read" access
3. Accept model terms at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

---

---

# GUI Setup (Web Interface)

**Prerequisites**: CLI must be installed first (see above).

## Additional Requirements

- **Node.js 18+**: [Download from nodejs.org](https://nodejs.org/)
- **npm 9+**: Comes with Node.js

### Verify Node.js Installation

```bash
node --version  # Should be 18+
npm --version   # Should be 9+
```

## Step 1: Navigate to GUI Backend

```bash
cd gui-app/backend
```

## Step 2: Activate CLI Virtual Environment

**Important**: The GUI backend shares the CLI's virtual environment to access core processing modules.

```bash
# Windows PowerShell
..\..\audio-transcription-cli\venv\Scripts\Activate.ps1

# Windows CMD
..\..\audio-transcription-cli\venv\Scripts\activate.bat

# macOS/Linux
source ../../audio-transcription-cli/venv/bin/activate
```

## Step 3: Install GUI Backend Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic (data validation)
- python-multipart (file uploads)

## Step 4: Configure GUI Backend

Create `.env` file in `gui-app/backend/`:

```bash
# Windows PowerShell
@"
HF_TOKEN=hf_YourActualTokenHere
CUDA_VISIBLE_DEVICES=0
API_HOST=0.0.0.0
API_PORT=8000
"@ | Out-File -FilePath .env -Encoding UTF8

# macOS/Linux
cat > .env << EOF
HF_TOKEN=hf_YourActualTokenHere
CUDA_VISIBLE_DEVICES=0
API_HOST=0.0.0.0
API_PORT=8000
EOF
```

## Step 5: Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

This installs:
- Next.js 15
- React 19
- TypeScript
- Tailwind CSS v3.4.14
- Lucide React (icons)

## Step 6: Configure Frontend

```bash
# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

---

## Testing

### Test 1: CLI Commands

```bash
# Activate CLI venv first
cd audio-transcription-cli
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate  # macOS/Linux

# Should show help
audio-transcription --help

# Test transcription command
audio-transcription transcribe --help

# Test diarization command
audio-transcription diarize --help
```

### Test 2: CLI Transcription

```bash
# Basic transcription (no diarization)
audio-transcription transcribe audio/sample_16k.wav --model large-v3 --language en
```

### Test 3: CLI Diarization

```bash
# Full diarization pipeline
audio-transcription diarize audio/sample_16k.wav --model large-v3 --min-speakers 2 --max-speakers 4 --language en
```

### Test 4: GUI Backend

**Terminal 1 - Start Backend**:
```bash
cd gui-app/backend
# Activate CLI venv
..\..\audio-transcription-cli\venv\Scripts\Activate.ps1  # Windows
# source ../../audio-transcription-cli/venv/bin/activate  # macOS/Linux

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected output**:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Test health endpoint** (in another terminal):
```bash
curl http://localhost:8000/health
# Should return JSON with status: healthy
```

**View API docs**: Open `http://localhost:8000/docs` in your browser

### Test 5: GUI Frontend

**Terminal 2 - Start Frontend**:
```bash
cd gui-app/frontend
npm run dev
```

**Expected output**:
```
▲ Next.js 15.x.x
- Local:        http://localhost:3000
- Ready in <time>
```

**Access the GUI**: Open `http://localhost:3000` in your browser

**Expected**:
- Black background
- White text clearly visible
- File upload area
- Configuration options

### Test 6: Full GUI Processing

1. Upload an audio file (drag and drop or click)
2. Configure options:
   - Model: large-v3
   - Language: en (or auto-detect)
   - Enable diarization
   - Min speakers: 2
   - Max speakers: 4
3. Click "Start Processing"
4. Watch real-time progress bar
5. View results when complete
6. Download output files

**Expected**:
- Real-time progress updates
- Processing completes successfully
- Results page shows speaker segments
- All download buttons work

---

## Troubleshooting

### CLI Issues

**Python version error**:
```bash
# Ensure Python 3.9+ is installed
python --version

# If needed, create venv with specific Python version
python3.10 -m venv venv
```

**CUDA not available**:
```bash
# Check CUDA
nvidia-smi

# If no GPU, use CPU mode
audio-transcription transcribe audio.wav --device cpu
```

**HuggingFace token issues**:
```bash
# Verify token is set
echo $env:HF_TOKEN  # Windows PowerShell
echo $HF_TOKEN      # macOS/Linux

# Accept model terms at:
# https://huggingface.co/pyannote/speaker-diarization-3.1
```

### GUI Issues

**Backend won't start - Import errors**:
```bash
# Ensure using CLI's venv, not a separate one
cd gui-app/backend
..\..\audio-transcription-cli\venv\Scripts\Activate.ps1  # Windows

# Verify audio_transcription module is available
python -c "import audio_transcription; print('OK')"
```

**Cannot access backend at 0.0.0.0:8000**:
```
Use http://localhost:8000 instead
(0.0.0.0 is for server binding, not browser access)
```

**Pydantic validation errors**:
```bash
# Check gui-app/backend/config.py has:
# - All .env variables defined as fields
# - extra = "ignore" in Config class
```

**Frontend won't start - npm errors**:
```bash
# Clear npm cache and reinstall
cd gui-app/frontend
rm -rf node_modules package-lock.json
npm install
```

**TypeScript CSS import errors**:
```bash
# Ensure global.d.ts exists in frontend directory
# Ensure tsconfig.json includes it
```

**UI text not visible**:
```
- Clear browser cache (Ctrl+Shift+Delete)
- Hard refresh (Ctrl+F5)
- Check for black background and white text
```

**Processing fails / KeyError: 'filename'**:
```bash
# Check gui-app/backend/services/processor.py
# Session dict must include "filename": audio_path.name
```

### Getting More Help

- **CLI Detailed Docs**: See `README.md` → Troubleshooting section
- **GUI Detailed Setup**: See `gui-app/QUICKSTART.md`
- **Implementation Details**: See `.claude/deployment.md`
- **Report Issues**: https://github.com/TharanaBope/whisper-v3-diarization/issues

---

## ✅ Setup Complete!

You're ready to use the audio transcription system!

### Quick Reference

**CLI Usage**:
```bash
# Activate venv
cd audio-transcription-cli
.\venv\Scripts\Activate.ps1  # Windows

# Run transcription
audio-transcription process audio/file.wav --model large-v3 --language en
```

**GUI Usage**:
```bash
# Terminal 1: Backend
cd gui-app/backend
..\..\audio-transcription-cli\venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd gui-app/frontend
npm run dev

# Browser: http://localhost:3000
```

**Need help?** See the main [README.md](README.md) for detailed documentation.
