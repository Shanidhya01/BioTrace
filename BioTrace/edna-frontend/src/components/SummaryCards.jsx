import React from "react";

const SummaryCard = ({ data }) => {
  if (!data || data.length === 0) return null;

  const total = data.length;
  const uniqueTaxa = new Set(data.map((row) => row.Taxonomy || "Unknown")).size;

  // Calculate average confidence
  const avgConfidence = data.reduce((sum, row) => sum + (row.Confidence || 0), 0) / total;

  // Most common taxonomy
  const counts = data.reduce((acc, row) => {
    const key = row.Taxonomy || "Unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const mostCommon = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
      <div className="bg-white p-4 rounded-2xl shadow-md text-center">
        <h2 className="text-gray-500 text-sm">Total Records</h2>
        <p className="text-2xl font-bold">{total}</p>
      </div>
      <div className="bg-white p-4 rounded-2xl shadow-md text-center">
        <h2 className="text-gray-500 text-sm">Unique Taxa</h2>
        <p className="text-2xl font-bold">{uniqueTaxa}</p>
      </div>
      <div className="bg-white p-4 rounded-2xl shadow-md text-center">
        <h2 className="text-gray-500 text-sm">Most Common Taxon</h2>
        <p className="text-lg font-semibold">{mostCommon?.[0] || "N/A"}</p>
        <p className="text-sm text-gray-500">{mostCommon?.[1]} records</p>
      </div>
      <div className="bg-white p-4 rounded-2xl shadow-md text-center">
        <h2 className="text-gray-500 text-sm">Avg Confidence</h2>
        <p className="text-2xl font-bold">{(avgConfidence * 100).toFixed(1)}%</p>
      </div>
    </div>
  );
};

export default SummaryCard;