import React, { useRef } from "react";
import { Button } from "./ui/button";
import { uploadCSV } from "../api/api";

const UploadForm = ({ onUpload }) => {
  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv')) {
      alert('Please select a CSV file (extension .csv)');
      return;
    }

    try {
      console.log("Attempting to upload file:", file.name, "size:", file.size);
      
      // Create FormData properly
      const formData = new FormData();
      formData.append('file', file); // 'file' must match the FastAPI parameter name

      // Upload using FormData
      const response = await uploadCSV(formData);
      console.log("Upload response:", response);
      
      if (response.data) {
        onUpload(response.data);
      } else {
        throw new Error("No data received from server");
      }
    } catch (err) {
      console.error("Upload failed:", err);
      // alert(`Upload failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-2">
      <input
        type="file"
        accept=".csv"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileChange}
      />
      <Button onClick={() => fileInputRef.current.click()} className="bg-blue-600 hover:bg-blue-700 text-white">
        Upload CSV
      </Button>
      <p className="text-sm text-gray-500">Upload a CSV file with DNA sequences</p>
    </div>
  );
};

export default UploadForm;