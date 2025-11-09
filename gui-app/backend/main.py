"""FastAPI application entry point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes import upload, process, results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Audio Transcription & Diarization API",
    description="Backend API for audio transcription with speaker diarization",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(process.router)
app.include_router(results.router)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Audio Transcription API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "storage": {
            "uploads_dir": str(settings.UPLOAD_DIR),
            "results_dir": str(settings.RESULTS_DIR)
        },
        "config": {
            "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
            "allowed_extensions": list(settings.ALLOWED_EXTENSIONS),
            "default_model": settings.DEFAULT_MODEL
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
