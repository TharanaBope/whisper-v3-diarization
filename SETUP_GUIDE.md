# Audio Transcription & Diarization CLI - Complete Setup Guide

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

## Testing the Application

```bash
# Should show help
audio-transcription --help

# Test transcription command
audio-transcription transcribe --help

# Test diarization command
audio-transcription diarize --help
```

### Test 2: Transcribe Sample Audio

```bash
# Basic transcription (no diarization)
audio-transcription transcribe audio/sample_16k.wav --model large-v3 --language en
```

### Test 3: Diarization with Speaker Detection

```bash
# Full diarization pipeline
audio-transcription diarize audio/sample_16k.wav --model large-v3 --min-speakers 2 --max-speakers 4 --language en
```

---
