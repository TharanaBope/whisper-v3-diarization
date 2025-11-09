"use client";

import { Download, Users } from "lucide-react";
import { ProcessingResult } from "@/lib/types";
import { getDownloadUrl } from "@/lib/api";

interface ResultsViewerProps {
  result: ProcessingResult;
}

export default function ResultsViewer({ result }: ResultsViewerProps) {
  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg p-6 text-black border border-gray-700">
        <h2 className="text-2xl font-bold mb-2">Processing Complete!</h2>
        <p className="text-gray-700">{result.filename}</p>
      </div>

      {/* Stats */}
      {result.diarization && (
        <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-md p-6">
          <div className="flex items-center gap-3 mb-4">
            <Users className="h-6 w-6 text-white" />
            <h3 className="text-lg font-semibold text-white">Speaker Detection</h3>
          </div>
          <p className="text-gray-300">
            <span className="font-bold text-2xl text-white">
              {result.diarization.num_speakers}
            </span>{" "}
            speakers detected
          </p>
          <div className="flex flex-wrap gap-2 mt-3">
            {result.diarization.speakers?.map((speaker) => (
              <span
                key={speaker}
                className="px-3 py-1 bg-gray-800 text-white border border-gray-700 rounded-full text-sm"
              >
                {speaker}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Transcription Preview */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4 text-white">Transcription</h3>
        <div className="bg-gray-800 rounded-lg p-4 max-h-96 overflow-y-auto">
          {result.diarization ? (
            <div className="space-y-3">
              {result.diarization.segments?.slice(0, 10).map((segment, index) => (
                <div key={index} className="border-l-4 border-white pl-4">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-bold text-white">
                      {segment.speaker}
                    </span>
                    <span className="text-xs text-gray-400">
                      {Math.floor(segment.start)}s - {Math.floor(segment.end)}s
                    </span>
                  </div>
                  <p className="text-sm text-gray-300">{segment.text}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {result.transcription?.segments?.slice(0, 10).map((segment, index) => (
                <div key={index}>
                  <span className="text-xs text-gray-400 mr-2">
                    {Math.floor(segment.start)}s
                  </span>
                  <span className="text-sm text-gray-300">{segment.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Download Buttons */}
      <div className="bg-gray-900 border border-gray-700 rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white">
          <Download className="h-5 w-5" />
          Download Results
        </h3>
        <div className="grid grid-cols-2 gap-4">
          {result.transcription_txt_url && (
            <a
              href={getDownloadUrl(result.session_id, "transcription.txt")}
              download
              className="px-4 py-2 bg-white text-black rounded-lg hover:bg-gray-200 transition text-center"
            >
              Transcription (TXT)
            </a>
          )}
          {result.transcription_json_url && (
            <a
              href={getDownloadUrl(result.session_id, "transcription.json")}
              download
              className="px-4 py-2 bg-white text-black rounded-lg hover:bg-gray-200 transition text-center"
            >
              Transcription (JSON)
            </a>
          )}
          {result.diarization_txt_url && (
            <a
              href={getDownloadUrl(result.session_id, "diarization.txt")}
              download
              className="px-4 py-2 bg-white text-black rounded-lg hover:bg-gray-200 transition text-center"
            >
              Diarization (TXT)
            </a>
          )}
          {result.diarization_json_url && (
            <a
              href={getDownloadUrl(result.session_id, "diarization.json")}
              download
              className="px-4 py-2 bg-white text-black rounded-lg hover:bg-gray-200 transition text-center"
            >
              Diarization (JSON)
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
