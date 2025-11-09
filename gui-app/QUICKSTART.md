# GUI Quick Start Guide

Complete setup and usage instructions for the Audio Transcription GUI.

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Backend Dependencies

```bash
# From transcribe-backend directory
cd audio-transcription-cli

# Activate the venv (create one if needed)
.\venv\Scripts\activate   # Windows
# OR
source venv/bin/activate  # Linux/Mac

# Install FastAPI dependencies
cd ../gui-app/backend
pip install -r requirements.txt
```

### Step 2: Configure Backend

```bash
# Create environment file
cp .env.example .env

# Edit .env and add your HuggingFace token
# Use notepad, vim, or any editor:
notepad .env  # Windows
```

Add your token:
```
HF_TOKEN=your_actual_token_here
CUDA_VISIBLE_DEVICES=0
API_HOST=0.0.0.0
API_PORT=8000
```

### Step 3: Install Frontend Dependencies

```bash
cd ../frontend
npm install
```

### Step 4: Configure Frontend

```bash
cp .env.local.example .env.local
# Default config should work (http://localhost:8000)
```

## ▶️ Running the Application

### Terminal 1: Start Backend

```bash
# From transcribe-backend/gui-app/backend/
cd ../../audio-transcription-cli
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

cd ../../gui-app/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Test it**: Open http://localhost:8000 - you should see `{"status":"healthy",...}`

### Terminal 2: Start Frontend

```bash
# From transcribe-backend/gui-app/frontend/
npm run dev
```

You should see:
```
  ▲ Next.js 15.0.0
  - Local:        http://localhost:3000

 ✓ Starting...
 ✓ Ready in 2.5s
```

**Open the app**: http://localhost:3000

## 📱 Using the Application

### 1. Upload Audio File
- Drag and drop an audio file OR click to browse
- Supported: MP3, WAV, FLAC, OGG, M4A, AAC (max 500MB)

### 2. Configure Processing
- **Model Size**: large-v3 (best quality) to tiny (fastest)
- **Language**: Optional (auto-detects if empty), e.g., `en`, `es`, `fr`
- **Device**: Auto (recommended), CUDA (GPU), or CPU
- **Distil-Whisper**: Enable for 2-5x faster processing
- **Speaker Diarization**:
  - Enable to identify different speakers
  - Optionally set min/max speaker count

### 3. Watch Progress
- Real-time progress bar and status updates
- Processing stages:
  - Initializing (5%)
  - Loading audio (10%)
  - Transcribing (30-60%)
  - Diarizing speakers (70-95%)
  - Complete (100%)

### 4. View Results
- See detected speakers with color-coded labels
- Read timestamped transcription
- Download results:
  - Transcription TXT/JSON
  - Diarization TXT/JSON (if enabled)

## 🎯 Example Workflow

```
1. Upload "meeting.wav" (100MB, 30 minutes)
2. Configure:
   - Model: large-v3
   - Language: en
   - Diarization: ON (2-4 speakers)
   - Distil-Whisper: ON
3. Processing time: ~5-10 minutes (with GPU)
4. Results:
   - 3 speakers detected (SPEAKER_00, SPEAKER_01, SPEAKER_02)
   - Full transcription with timestamps
   - Download all 4 result files
```

## ✅ Verify Everything Works

### Test Backend
```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","storage":{...},"config":{...}}
```

### Test Frontend
- Open http://localhost:3000
- Should see the upload page with blue-purple gradient background
- Drag-drop zone should be visible

### Test Full Flow
1. Upload a small test audio file (30 seconds recommended)
2. Use default settings
3. Click "Start Processing"
4. Watch progress stream
5. View results page
6. Download files

## 🔧 Troubleshooting

### Backend won't start
```bash
# Check if CLI venv is activated
which python   # Should show path to audio-transcription-cli/venv

# Verify CLI modules are installed
python -c "from audio_transcription.core.audio_processor import AudioProcessor"
# Should complete without errors
```

### Frontend won't start
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Import errors in backend
```bash
# Reinstall CLI
cd ../../audio-transcription-cli
pip install -e .
```

### CORS errors
- Ensure backend is running on port 8000
- Check `backend/config.py` has `CORS_ORIGINS = ["http://localhost:3000"]`

### SSE connection fails
- Verify backend URL in frontend `.env.local`
- Check browser console for errors
- Try different browser (Chrome/Firefox recommended)

## 📊 System Requirements

### Minimum
- Python 3.9+
- Node.js 20+
- 8GB RAM
- 10GB disk space

### Recommended
- Python 3.9+
- Node.js 20+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 20GB disk space
- CUDA 11.8+

## 🎓 Next Steps

After basic setup works:

1. **Test with real audio**: Try your own meeting recordings
2. **Experiment with settings**: Compare model sizes and speeds
3. **Try different languages**: Test multi-language support
4. **Batch processing**: Process multiple files (via results → upload another)
5. **Production deployment**: See `gui-implementation.md` for Docker setup

## 📚 Additional Resources

- **Full implementation guide**: `.claude/gui-implementation.md`
- **Backend setup details**: `backend/SETUP.md`
- **API documentation**: http://localhost:8000/docs (when backend running)
- **Deployment guide**: `.claude/deployment.md`
- **Code standards**: `.claude/rules.md`

## 🆘 Getting Help

If you encounter issues:

1. Check this QUICKSTART.md
2. Review `backend/SETUP.md` for backend issues
3. Check browser console for frontend errors
4. Verify both terminals show servers running
5. Ensure HuggingFace token is valid and terms accepted

---

**🎉 You're ready to go! Open http://localhost:3000 and start transcribing!**
