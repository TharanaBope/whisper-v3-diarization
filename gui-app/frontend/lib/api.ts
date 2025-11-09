/**
 * API client for backend communication
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function uploadAudioFile(
  file: File,
  options: {
    model_size: string;
    language?: string;
    enable_diarization: boolean;
    min_speakers?: number;
    max_speakers?: number;
    use_assistant: boolean;
    device: string;
  }
): Promise<{ session_id: string; filename: string; file_size: number; message: string }> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_size", options.model_size);
  if (options.language) formData.append("language", options.language);
  formData.append("enable_diarization", String(options.enable_diarization));
  if (options.min_speakers) formData.append("min_speakers", String(options.min_speakers));
  if (options.max_speakers) formData.append("max_speakers", String(options.max_speakers));
  formData.append("use_assistant", String(options.use_assistant));
  formData.append("device", options.device);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Upload failed");
  }

  return response.json();
}

export async function getResults(sessionId: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/results/${sessionId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Failed to fetch results");
  }

  return response.json();
}

export function getDownloadUrl(sessionId: string, fileType: string): string {
  return `${API_BASE_URL}/api/download/${sessionId}/${fileType}`;
}
