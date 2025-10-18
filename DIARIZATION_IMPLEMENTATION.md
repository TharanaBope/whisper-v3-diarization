# Speaker Diarization Implementation Guide

## Problem
OpenAI's Whisper model can transcribe audio but **cannot identify different speakers** (speaker diarization).

## Solution
Used **WhisperX** library to add speaker diarization capability on top of Whisper transcription.

---

## Implementation Steps

### 1. **Created CLI Application** ([audio-transcription-cli/](audio-transcription-cli/))
   - Built separate batch processing system
   - Two-stage approach: Transcribe → Diarize

### 2. **Transcription Stage** ([core/transcription.py](audio-transcription-cli/src/audio_transcription/core/transcription.py))
   - Uses standard Whisper model for transcription
   - Audio preprocessing: noise reduction, normalization
   - Chunked processing: 30s chunks with 5s overlap
   - Output: Plain text transcription

### 3. **Diarization Stage** ([core/diarization.py](audio-transcription-cli/src/audio_transcription/core/diarization.py))
   - Uses WhisperX library
   - **4-Step Pipeline**:
     1. **Transcribe**: WhisperX model generates text
     2. **Align**: wav2vec2 aligns words to audio timestamps
     3. **Diarize**: pyannote.audio identifies speaker segments
     4. **Assign**: Maps speakers to text segments
   - Output: Transcription with speaker labels (SPEAKER_00, SPEAKER_01, etc.)

### 4. **Fallback Mechanism**
   - WhisperX has compatibility issues with some audio
   - Implemented fallback: sentence-based speaker assignment
   - Uses standard transcription + time-based speaker rotation

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Transcription | OpenAI Whisper | Convert speech to text |
| Diarization | WhisperX + pyannote.audio | Identify speakers |
| Alignment | wav2vec2 | Precise word timing |
| Audio Processing | librosa, soundfile | Preprocessing & validation |

---

## How Diarization Works (Step-by-Step Process)

### **User Journey**

1. **Upload Audio File**
   - User provides audio file (MP3, WAV, FLAC, etc.)
   - Example: `audio/meeting_recording.wav`

2. **Audio Validation**
   - System checks file format and integrity
   - Converts to 16kHz sample rate (standard for speech processing)
   - Applies noise reduction and normalization

3. **Stage 1: Transcription** ([core/transcription.py:139-265](audio-transcription-cli/src/audio_transcription/core/transcription.py))
   - Whisper model converts speech to text
   - Processes in 30-second chunks with 5-second overlap
   - Merges chunks into full transcription
   - **Output**: `"Hello, welcome to our meeting. Thank you for having me..."`

4. **Stage 2: Diarization** ([core/diarization.py:72-184](audio-transcription-cli/src/audio_transcription/core/diarization.py))

   **Step 2a: Re-transcribe with WhisperX**
   - WhisperX transcribes audio again (more detailed than Stage 1)
   - Generates word-level timestamps
   - Example: `{"word": "Hello", "start": 0.5, "end": 0.8}`

   **Step 2b: Align Words to Audio** ([diarization.py:52-57](audio-transcription-cli/src/audio_transcription/core/diarization.py))
   - Uses **wav2vec2** alignment model
   - Precisely maps each word to exact audio position
   - Refines timestamps for accuracy
   - Code: `whisperx.load_align_model(language_code="en", device="cuda")`

   **Step 2c: Speaker Diarization with pyannote.audio** ([diarization.py:60-66](audio-transcription-cli/src/audio_transcription/core/diarization.py))
   - **What is pyannote.audio?**
     - Deep learning toolkit for speaker diarization
     - Trained on thousands of hours of conversational audio
     - Analyzes voice characteristics (pitch, tone, rhythm)
     - Identifies "who spoke when" without knowing speaker names

   - **How pyannote works:**
     1. Voice Activity Detection (VAD): Finds speech segments
     2. Speaker Embedding: Creates unique "voiceprint" for each speaker
     3. Clustering: Groups similar voiceprints together
     4. Assigns labels: SPEAKER_00, SPEAKER_01, SPEAKER_02, etc.

   - **Code Implementation:**
     ```python
     # Load pyannote diarization model
     self.diarize_model = whisperx.DiarizationPipeline(
         model_name="pyannote/speaker-diarization@2.1",
         use_auth_token=self.hf_token,  # Requires HuggingFace token
         device=self.device
     )

     # Run diarization
     diarize_segments = self.diarize_model(audio_path, min_speakers=2, max_speakers=4)
     ```

   - **Output from pyannote:**
     ```
     [0.0s - 5.2s] SPEAKER_00
     [5.5s - 10.8s] SPEAKER_01
     [11.0s - 15.3s] SPEAKER_00
     ```

   **Step 2d: Assign Speakers to Words**
   - Matches speaker segments with word timestamps
   - Each word gets assigned to a speaker
   - Combines text + speaker + timestamp

   **Final Output:**
   ```
   [00:00] SPEAKER_00: Hello, welcome to our meeting today.
   [00:05] SPEAKER_01: Thank you for having me, it's great to be here.
   [00:10] SPEAKER_00: Let's start by discussing the project timeline.
   ```

5. **Save Results**
   - JSON file: `diarizations/filename_diarization.json` (complete data)
   - Text file: `diarizations/filename_diarized_transcript.txt` (formatted)

---

### **Why WhisperX + pyannote.audio?**

| Feature | Standard Whisper | WhisperX + pyannote |
|---------|------------------|---------------------|
| Transcription | ✅ Yes | ✅ Yes |
| Word timestamps | ❌ Sentence-level only | ✅ Word-level precision |
| Speaker identification | ❌ No | ✅ Yes |
| Processing speed | Normal | 70x faster (batched) |
| Voice fingerprinting | ❌ No | ✅ Yes (pyannote) |

---

### **Technical Requirements**

1. **HuggingFace Token** ([.env](audio-transcription-cli/.env))
   - pyannote models require authentication
   - Get token from: https://huggingface.co/settings/tokens
   - Accept terms at: https://huggingface.co/pyannote/speaker-diarization
   - Set in `.env`: `HF_TOKEN=hf_your_token_here`

2. **GPU Recommended**
   - pyannote.audio is compute-intensive
   - CUDA-enabled GPU significantly faster
   - CPU mode available but slower

3. **Dependencies**
   - `whisperx` - Enhanced Whisper with diarization
   - `pyannote.audio` - Speaker diarization models
   - `torch` - Deep learning framework
   - `librosa` - Audio processing

---

## Output Format

**Without Diarization** (Original):
```
Hello, welcome to our meeting today. Thank you for having me...
```

**With Diarization** (New):
```
[00:00] SPEAKER_00: Hello, welcome to our meeting today.
[00:05] SPEAKER_01: Thank you for having me, it's great to be here.
[00:10] SPEAKER_00: Let's start by discussing the project timeline.
```

---

## Usage

```bash
# Standard transcription only
audio-transcription transcribe audio/sample.wav --model large-v3

# Full diarization (transcription + speaker identification)
audio-transcription diarize audio/sample.wav --model large-v3 --min-speakers 2 --max-speakers 4
```

---

## Architecture Summary

```
┌─────────────────────────────────────┐
│   Production System (Prod/app.py)   │
│   • Real-time transcription         │
│   • No diarization                  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  CLI System (audio-transcription)   │
│                                     │
│  ┌───────────────────────────────┐ │
│  │  1. Whisper Transcription     │ │
│  │     (core/transcription.py)   │ │
│  └───────────────────────────────┘ │
│            ↓                        │
│  ┌───────────────────────────────┐ │
│  │  2. WhisperX Diarization      │ │
│  │     (core/diarization.py)     │ │
│  │     • Align words              │ │
│  │     • Identify speakers        │ │
│  │     • Assign labels            │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
```
