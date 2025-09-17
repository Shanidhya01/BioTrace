import React, { useRef, useState } from "react";
import { Button } from "./ui/button";

const MAX_FILE_MB = 20;

const UploadForm = ({ onJobStart, onUploadComplete, onUploadStart }) => {
  const [file, setFile] = useState(null);
  const [fileInfo, setFileInfo] = useState(null); // {name,sizeKB,validHeader:boolean}
  const [previewLines, setPreviewLines] = useState([]); // first few lines
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const resetState = () => {
    setFile(null);
    setFileInfo(null);
    setPreviewLines([]);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const readPreviewAndValidate = async (selectedFile) => {
    // Read small chunk for header & first 5 lines
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const text = String(reader.result || "");
        const lines = text.split(/\r?\n/).slice(0, 6);
        const header = (lines[0] || "").trim().toLowerCase();
        const hasSequence = header.includes("sequence");
        setPreviewLines(lines);
        setFileInfo({
          name: selectedFile.name,
          sizeKB: Math.round(selectedFile.size / 1024),
          validHeader: hasSequence,
        });
        if (!hasSequence) {
          setError("CSV header should contain a 'sequence' column.");
        } else {
          setError("");
        }
        resolve();
      };
      reader.onerror = () => {
        setError("Could not read file.");
        resolve();
      };
      // Read only first 64KB
      reader.readAsText(selectedFile.slice(0, 64 * 1024));
    });
  };

  const acceptFile = async (selectedFile) => {
    if (!selectedFile) return;
    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a CSV file (.csv).");
      return;
    }
    const sizeMB = selectedFile.size / (1024 * 1024);
    if (sizeMB > MAX_FILE_MB) {
      setError(`File too large. Max ${MAX_FILE_MB} MB.`);
      return;
    }
    setError("");
    setFile(selectedFile);
    await readPreviewAndValidate(selectedFile);
  };

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files?.[0];
    await acceptFile(selectedFile);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const selectedFile = e.dataTransfer.files?.[0];
    await acceptFile(selectedFile);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setError("");

    // notify parent immediately to hide old results
    onUploadStart?.();

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:8000/api/upload-csv/", {
        method: "POST",
        body: formData,
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || "Upload failed");
      if (json.job_id && onJobStart) onJobStart(json.job_id);
    } catch (err) {
      console.error(err);
      setError(err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Hidden native input */}
      <input
        type="file"
        accept=".csv"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Dropzone */}
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        className={`rounded-xl border-2 p-6 transition
          ${dragActive ? "border-blue-500 bg-blue-50" : "border-dashed border-blue-300 bg-white/80"}
        `}
      >
        <div className="flex flex-col items-center text-center">
          <div className="text-4xl mb-2">📄</div>
          <p className="text-sm text-blue-900 font-medium">
            Drag & drop your CSV here, or
          </p>
          <Button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="mt-2 bg-blue-600 hover:bg-blue-700 text-white"
          >
            Choose CSV
          </Button>
          <p className="text-xs text-blue-700 mt-2">
            Only .csv files. Max {MAX_FILE_MB} MB. Must include a "sequence" column.
          </p>

          {/* Selected file details */}
          {fileInfo && (
            <div className="mt-4 w-full max-w-md text-left bg-white/70 border border-blue-200 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm text-blue-900">
                  <span className="font-semibold">{fileInfo.name}</span>{" "}
                  <span className="text-blue-600">({fileInfo.sizeKB} KB)</span>
                </div>
                <div
                  className={`text-xs px-2 py-0.5 rounded ${
                    fileInfo.validHeader
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-amber-100 text-amber-700"
                  }`}
                >
                  {fileInfo.validHeader ? "Header OK" : "Missing 'sequence'"}
                </div>
              </div>

              {/* Preview first lines */}
              {!!previewLines.length && (
                <div className="mt-2 max-h-28 overflow-auto rounded bg-blue-50 border border-blue-200 p-2 text-[11px] font-mono text-blue-900">
                  {previewLines.map((l, i) => (
                    <div key={i}>{l}</div>
                  ))}
                </div>
              )}

              <div className="mt-3 flex gap-2">
                <button
                  type="button"
                  onClick={resetState}
                  className="px-3 py-1.5 text-xs rounded border border-blue-300 bg-white text-blue-700 hover:bg-blue-50"
                >
                  Clear
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Helper text / error */}
      {error ? (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg p-2">
          {error}
        </div>
      ) : (
        <p className="text-sm text-gray-500">
          Upload a CSV file with DNA sequences to start prediction.
        </p>
      )}

      {/* Actions */}
      <div className="flex items-center justify-center gap-3 w-full">
        <button
          type="submit"
          disabled={!file || uploading}
          className={`px-4 py-2 rounded-lg text-white text-sm font-semibold flex items-center gap-2 ${
            uploading ? "bg-blue-400" : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {uploading ? (
            <>
              <span className="inline-block h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
              Uploading…
            </>
          ) : (
            "Upload & Start Prediction"
          )}
        </button>
      </div>
    </form>
  );
};

export default UploadForm;