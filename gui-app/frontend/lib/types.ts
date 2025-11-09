/**
 * TypeScript types for API communication
 */

export interface UploadResponse {
  session_id: string;
  filename: string;
  file_size: number;
  message: string;
}

export interface ProcessingOptions {
  model_size: string;
  language?: string;
  enable_diarization: boolean;
  min_speakers?: number;
  max_speakers?: number;
  use_assistant: boolean;
  device: string;
}

export type ProgressEventType = "progress" | "complete" | "error";

export interface ProgressEvent {
  event_type: ProgressEventType;
  session_id: string;
  stage?: string;
  progress?: number;
  message?: string;
  error?: string;
  results_available?: boolean;
}

export interface ProcessingResult {
  session_id: string;
  success: boolean;
  filename: string;
  transcription?: {
    text: string;
    segments: Array<{
      start: number;
      end: number;
      text: string;
    }>;
  };
  diarization?: {
    num_speakers: number;
    speakers: string[];
    segments: Array<{
      start: number;
      end: number;
      text: string;
      speaker: string;
    }>;
  };
  transcription_txt_url?: string;
  transcription_json_url?: string;
  diarization_txt_url?: string;
  diarization_json_url?: string;
  error?: string;
}
