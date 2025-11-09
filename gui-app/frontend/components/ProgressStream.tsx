"use client";

import { useEffect, useState } from "react";
import { Loader2, CheckCircle, XCircle } from "lucide-react";
import { ProgressEvent } from "@/lib/types";

interface ProgressStreamProps {
  sessionId: string;
  onComplete: () => void;
  onError: (error: string) => void;
}

const STAGE_LABELS: Record<string, string> = {
  initializing: "Initializing processor",
  loading_audio: "Loading audio file",
  transcribing: "Transcribing audio",
  transcription_complete: "Transcription complete",
  diarizing: "Identifying speakers",
  diarization_complete: "Speaker identification complete",
  complete: "Processing complete",
  error: "Error occurred"
};

export default function ProgressStream({ sessionId, onComplete, onError }: ProgressStreamProps) {
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    const eventSource = new EventSource(
      `http://localhost:8000/api/progress/${sessionId}`
    );

    eventSource.onmessage = (event) => {
      const progressEvent: ProgressEvent = JSON.parse(event.data);

      setEvents((prev) => [...prev, progressEvent]);

      if (progressEvent.progress !== undefined) {
        setCurrentProgress(progressEvent.progress);
      }

      if (progressEvent.stage) {
        setCurrentStage(STAGE_LABELS[progressEvent.stage] || progressEvent.stage);
      }

      if (progressEvent.event_type === "complete") {
        setIsComplete(true);
        eventSource.close();
        onComplete();
      } else if (progressEvent.event_type === "error") {
        setHasError(true);
        eventSource.close();
        onError(progressEvent.error || "Unknown error");
      }
    };

    eventSource.onerror = (error) => {
      console.error("EventSource error:", error);
      setHasError(true);
      eventSource.close();
      onError("Connection to server lost");
    };

    return () => {
      eventSource.close();
    };
  }, [sessionId, onComplete, onError]);

  return (
    <div className="w-full space-y-6">
      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-white">
            {currentStage || "Starting..."}
          </span>
          <span className="text-sm text-gray-300">{currentProgress}%</span>
        </div>

        <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full transition-all duration-300 rounded-full ${
              hasError ? "bg-red-500" : isComplete ? "bg-white" : "bg-white"
            }`}
            style={{ width: `${currentProgress}%` }}
          />
        </div>
      </div>

      {/* Status Icon */}
      <div className="flex items-center justify-center">
        {hasError ? (
          <XCircle className="h-16 w-16 text-red-500" />
        ) : isComplete ? (
          <CheckCircle className="h-16 w-16 text-white" />
        ) : (
          <Loader2 className="h-16 w-16 text-white animate-spin" />
        )}
      </div>

      {/* Event Log */}
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 max-h-64 overflow-y-auto">
        <h3 className="text-sm font-semibold text-white mb-2">Processing Log</h3>
        <div className="space-y-1">
          {events.map((event, index) => (
            <div key={index} className="text-xs text-gray-300 font-mono">
              <span className="text-gray-500">[{event.progress || 0}%]</span> {event.message}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
