import React from "react";

const DataTable = ({ data }) => {
  if (!data || data.length === 0) return null;

  return (
    <div className="bg-white p-4 rounded-2xl shadow-md">
      <h2 className="text-lg font-bold mb-3">Raw Data</h2>
      <div className="overflow-x-auto">
        <table className="w-full border">
          <thead>
            <tr className="bg-gray-100">
              <th className="border px-4 py-2 text-left">Sequence</th>
              <th className="border px-4 py-2 text-left">Predicted Species</th>
              <th className="border px-4 py-2 text-left">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i} className="hover:bg-gray-50">
                <td className="border px-4 py-2 font-mono text-sm">
                  {row.sequence.length > 50 ? row.sequence.substring(0, 50) + '...' : row.sequence}
                </td>
                <td className="border px-4 py-2 font-medium">{row.Taxonomy}</td>
                <td className="border px-4 py-2">{((row.Confidence || 0) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DataTable;