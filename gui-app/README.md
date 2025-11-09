# Audio Transcription GUI Application

Web-based interface for audio transcription and speaker diarization using Next.js 15 and FastAPI.

## Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS v3.4.14
- **Backend**: FastAPI, Python 3.9+
- **Processing**: OpenAI Whisper, WhisperX, pyannote-audio

## Prerequisites

- Python 3.9+ with pip
- Node.js 20+ with npm
- HuggingFace account and token
- (Optional) NVIDIA GPU with CUDA for faster processing

## Installation

### 1. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit .env and add your HuggingFace token
# HF_TOKEN=your_token_here
```

### 2. Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Create environment file
cp .env.local.example .env.local
```

## Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

Frontend will be available at: http://localhost:3000

## Usage

1. **Upload Audio**: Drag-drop or click to select audio file (MP3, WAV, FLAC, OGG, M4A, AAC)
2. **Configure Options**:
   - Model size (tiny to large-v3)
   - Language (optional, auto-detects if empty)
   - Device (auto/GPU/CPU)
   - Distil-Whisper assistant (2-5x faster)
   - Speaker diarization settings
3. **Process**: Click "Start Processing" and watch live progress
4. **View Results**: See transcription, speaker labels, and download files

## Features

- ✅ Drag-and-drop file upload
- ✅ Real-time progress streaming via SSE
- ✅ Speaker diarization (2-10 speakers)
- ✅ Multiple model sizes
- ✅ Distil-Whisper for faster processing
- ✅ Download results (JSON + TXT)
- ✅ GPU acceleration support

## File Size Limits

- Maximum file size: 500MB
- Supported formats: .mp3, .wav, .flac, .ogg, .m4a, .aac

## Troubleshooting

### Backend Issues

**ModuleNotFoundError: audio_transcription**
```bash
# Ensure CLI module is installed
cd ../../audio-transcription-cli
pip install -e .
```

**HuggingFace token error**
- Get token from: https://huggingface.co/settings/tokens
- Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1
- Add to backend/.env file

### Frontend Issues

**CORS errors**
- Verify backend CORS_ORIGINS in backend/config.py includes "http://localhost:3000"

**SSE connection fails**
- Check backend is running on port 8000
- Clear browser cache and reload

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload audio file |
| GET | `/api/progress/{session_id}` | SSE progress stream |
| GET | `/api/results/{session_id}` | Get final results |
| GET | `/api/download/{session_id}/{file_type}` | Download result files |

## Development

### Frontend Development

```bash
cd frontend
npm run dev    # Development server
npm run build  # Production build
npm run lint   # Run ESLint
```

### Backend Development

```bash
cd backend
# Edit files, uvicorn auto-reloads with --reload flag
```

## Version Compatibility

- **Next.js**: 15.0.0 (LTS)
- **React**: 19.0.0
- **Tailwind CSS**: 3.4.14 (NOT v4 - has compatibility issues with Next.js 15)
- **TypeScript**: 5.6.0
- **FastAPI**: 0.115.0
- **Python**: 3.9+

## Project Structure

```
gui-app/
├── backend/
│   ├── main.py                # FastAPI app
│   ├── config.py              # Settings
│   ├── models.py              # Pydantic models
│   ├── routes/                # API endpoints
│   ├── services/              # Processing logic
│   └── storage/               # Uploads & results
│
└── frontend/
    ├── app/                   # Next.js pages
    ├── components/            # React components
    └── lib/                   # API client & types
```

## License

Same as parent project.

## Support

For issues and questions, refer to:
- Project documentation: `../.claude/gui-implementation.md`
- Deployment guide: `../.claude/deployment.md`
- Code standards: `../.claude/rules.md`
