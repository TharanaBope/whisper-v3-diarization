"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import FileUpload from "@/components/FileUpload";
import ProgressStream from "@/components/ProgressStream";
import { uploadAudioFile } from "@/lib/api";
import { Settings } from "lucide-react";

type Step = "upload" | "configure" | "processing" | "complete";

export default function Home() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string>("");
  const [isUploading, setIsUploading] = useState(false);

  // Configuration options
  const [modelSize, setModelSize] = useState("large-v3");
  const [language, setLanguage] = useState("");
  const [enableDiarization, setEnableDiarization] = useState(true);
  const [minSpeakers, setMinSpeakers] = useState<number | undefined>(undefined);
  const [maxSpeakers, setMaxSpeakers] = useState<number | undefined>(undefined);
  const [useAssistant, setUseAssistant] = useState(false);
  const [device, setDevice] = useState("auto");

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setStep("configure");
  };

  const handleStartProcessing = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    try {
      const response = await uploadAudioFile(selectedFile, {
        model_size: modelSize,
        language: language || undefined,
        enable_diarization: enableDiarization,
        min_speakers: minSpeakers,
        max_speakers: maxSpeakers,
        use_assistant: useAssistant,
        device: device
      });

      setSessionId(response.session_id);
      setStep("processing");
    } catch (error) {
      alert(`Upload failed: ${error}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleComplete = () => {
    router.push(`/results/${sessionId}`);
  };

  const handleError = (error: string) => {
    alert(`Processing failed: ${error}`);
    setStep("upload");
    setSelectedFile(null);
  };

  const handleReset = () => {
    setStep("upload");
    setSelectedFile(null);
    setSessionId("");
  };

  return (
    <div className="min-h-screen bg-black py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            Audio Transcription & Diarization
          </h1>
          <p className="text-lg text-gray-300">
            Upload your audio file and get accurate transcription with speaker identification
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-gray-900 border border-gray-700 rounded-2xl shadow-xl p-8">
          {step === "upload" && (
            <div>
              <h2 className="text-2xl font-semibold mb-6 text-white">Step 1: Upload Audio File</h2>
              <FileUpload onFileSelect={handleFileSelect} />
            </div>
          )}

          {step === "configure" && selectedFile && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-semibold text-white">Step 2: Configure Processing</h2>
                <button
                  onClick={handleReset}
                  className="text-sm text-gray-400 hover:text-gray-200"
                >
                  Change file
                </button>
              </div>

              {/* File Info */}
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
                <p className="text-sm font-medium text-gray-300">Selected file:</p>
                <p className="text-lg font-semibold text-white">{selectedFile.name}</p>
              </div>

              {/* Configuration Form */}
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Model Size
                  </label>
                  <select
                    value={modelSize}
                    onChange={(e) => setModelSize(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 text-white rounded-lg focus:ring-2 focus:ring-white"
                  >
                    <option value="large-v3">Large V3 (Best Quality)</option>
                    <option value="large-v2">Large V2</option>
                    <option value="medium">Medium</option>
                    <option value="small">Small</option>
                    <option value="base">Base</option>
                    <option value="tiny">Tiny (Fastest)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Language (Optional)
                  </label>
                  <input
                    type="text"
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    placeholder="en, es, fr, etc. (auto-detect if empty)"
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 text-white placeholder-gray-500 rounded-lg focus:ring-2 focus:ring-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Device
                  </label>
                  <select
                    value={device}
                    onChange={(e) => setDevice(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 text-white rounded-lg focus:ring-2 focus:ring-white"
                  >
                    <option value="auto">Auto (Recommended)</option>
                    <option value="cuda">GPU (CUDA)</option>
                    <option value="cpu">CPU</option>
                  </select>
                </div>

                <div className="flex items-center">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useAssistant}
                      onChange={(e) => setUseAssistant(e.target.checked)}
                      className="w-5 h-5 rounded focus:ring-2 focus:ring-white bg-gray-800 border-gray-700"
                    />
                    <span className="text-sm font-medium text-gray-300">
                      Use Distil-Whisper (2-5x faster)
                    </span>
                  </label>
                </div>
              </div>

              {/* Diarization Settings */}
              <div className="border-t border-gray-700 pt-6">
                <label className="flex items-center gap-2 mb-4 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableDiarization}
                    onChange={(e) => setEnableDiarization(e.target.checked)}
                    className="w-5 h-5 rounded focus:ring-2 focus:ring-white bg-gray-800 border-gray-700"
                  />
                  <span className="text-sm font-medium text-gray-300">
                    Enable Speaker Diarization (Identify different speakers)
                  </span>
                </label>

                {enableDiarization && (
                  <div className="grid grid-cols-2 gap-6 ml-7">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Min Speakers (Optional)
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={minSpeakers || ""}
                        onChange={(e) => setMinSpeakers(e.target.value ? parseInt(e.target.value) : undefined)}
                        placeholder="Auto-detect"
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 text-white placeholder-gray-500 rounded-lg focus:ring-2 focus:ring-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Max Speakers (Optional)
                      </label>
                      <input
                        type="number"
                        min="1"
                        value={maxSpeakers || ""}
                        onChange={(e) => setMaxSpeakers(e.target.value ? parseInt(e.target.value) : undefined)}
                        placeholder="Auto-detect"
                        className="w-full px-4 py-2 bg-gray-800 border border-gray-700 text-white placeholder-gray-500 rounded-lg focus:ring-2 focus:ring-white"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Start Button */}
              <button
                onClick={handleStartProcessing}
                disabled={isUploading}
                className="w-full py-4 bg-white text-black font-semibold rounded-lg hover:bg-gray-200 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isUploading ? "Uploading..." : "Start Processing"}
              </button>
            </div>
          )}

          {step === "processing" && sessionId && (
            <div>
              <h2 className="text-2xl font-semibold mb-6 text-white">Processing Your Audio</h2>
              <ProgressStream
                sessionId={sessionId}
                onComplete={handleComplete}
                onError={handleError}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
