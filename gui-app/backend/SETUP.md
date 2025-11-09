# Backend Setup Guide

## Important: Use CLI Virtual Environment

The backend **shares the same virtual environment** as the CLI application to avoid duplicate dependencies.

### Installation Steps

1. **Ensure CLI dependencies are installed** (if not already done):

```bash
cd ../../audio-transcription-cli
pip install -r requirements.txt
pip install -e .
```

2. **Install additional backend-only dependencies**:

```bash
# Activate the CLI venv
cd audio-transcription-cli
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows

# Install FastAPI dependencies
cd ../gui-app/backend
pip install -r requirements.txt
```

3. **Create environment file**:

```bash
cp .env.example .env
```

4. **Edit `.env` and add your HuggingFace token**:

```
HF_TOKEN=your_huggingface_token_here
```

### Running the Backend

**Always use the CLI venv when running the backend:**

```bash
# From transcribe-backend/gui-app/backend/

# Windows
cd ../../audio-transcription-cli
.\venv\Scripts\activate
cd ../../gui-app/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Linux/Mac
cd ../../audio-transcription-cli
source venv/bin/activate
cd ../../gui-app/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Why Share the Venv?

The backend imports and uses the CLI core modules directly:
- `audio_transcription.core.audio_processor`
- `audio_transcription.core.diarization`
- `audio_transcription.utils.*`

Using the same venv ensures:
- ✅ No duplicate PyTorch/Whisper installations (saves ~5GB disk space)
- ✅ Consistent model versions between CLI and GUI
- ✅ Same CUDA configuration
- ✅ Easier dependency management

### Verify Setup

Test that backend can import CLI modules:

```bash
# With CLI venv activated
cd gui-app/backend
python -c "import sys; sys.path.insert(0, '../../audio-transcription-cli/src'); from audio_transcription.core.audio_processor import AudioProcessor; print('Success!')"
```

If this works, your setup is correct!

### Troubleshooting

**Import errors**:
- Make sure you're using the CLI venv
- Verify CLI is installed: `cd audio-transcription-cli && pip install -e .`

**Module not found**:
- The backend's `services/processor.py` adds the CLI to Python path automatically
- Verify path: `../../audio-transcription-cli/src` exists

**HuggingFace errors**:
- Ensure `HF_TOKEN` is set in `.env`
- Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
