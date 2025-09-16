import axios from "axios";

const API_URL = "http://localhost:8000/api";

export const uploadCSV = async (formData) => {
  return await axios.post(`${API_URL}/upload-csv/`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};
