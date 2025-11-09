"use client";

import { useState, useRef, DragEvent } from "react";
import { Upload, File } from "lucide-react";

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function FileUpload({ onFileSelect, disabled }: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    const validExtensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"];
    const fileExt = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();

    if (!validExtensions.includes(fileExt)) {
      alert(`Invalid file format. Please upload: ${validExtensions.join(", ")}`);
      return;
    }

    if (file.size > 500 * 1024 * 1024) {
      alert("File too large. Maximum size is 500MB");
      return;
    }

    setSelectedFile(file);
    onFileSelect(file);
  };

  const onButtonClick = () => {
    inputRef.current?.click();
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept=".mp3,.wav,.flac,.ogg,.m4a,.aac"
        onChange={handleChange}
        disabled={disabled}
      />

      <div
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200
          ${dragActive ? "border-white bg-gray-800" : "border-gray-600 bg-gray-800"}
          ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-gray-400"}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={!disabled ? onButtonClick : undefined}
      >
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />

        {selectedFile ? (
          <div className="space-y-2">
            <div className="flex items-center justify-center gap-2">
              <File className="h-5 w-5 text-white" />
              <p className="text-sm font-medium text-white">{selectedFile.name}</p>
            </div>
            <p className="text-xs text-gray-400">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>
        ) : (
          <div>
            <p className="text-lg font-medium text-white mb-2">
              Drop audio file here or click to browse
            </p>
            <p className="text-sm text-gray-400">
              Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC (max 500MB)
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
