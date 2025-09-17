import React, { useState, useEffect, useMemo } from "react";
import { CSVLink } from "react-csv";

const PAGE_SIZE = 5;

const TaxonomyTable = ({ data }) => {
  const [sortField, setSortField] = useState("count");
  const [sortDirection, setSortDirection] = useState("desc");
  const [searchTerm, setSearchTerm] = useState("");
  const [activeRow, setActiveRow] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [page, setPage] = useState(0);

  useEffect(() => setIsLoaded(true), []);

  // --- NORMALIZE BACKEND DATA (predicted_species, confidence, etc.) ---
  const normalized = useMemo(() => {
    const rows = Array.isArray(data)
      ? data
      : (data && Array.isArray(data.results) ? data.results : []);

    return rows.map(r => {
      const taxonomy =
        r.predicted_species ||
        r.species ||
        r.taxon ||
        r.Taxonomy ||
        r.name ||
        "Unknown";

      const rawC = r.confidence ??
                   r.confidence_score ??
                   r.Confidence ??
                   r.score ??
                   0;

      // Convert to percentage (assume 0–1 incoming -> *100)
      let confidence = typeof rawC === "number" ? rawC : parseFloat(rawC) || 0;
      if (confidence <= 1) confidence = confidence * 100;
      return {
        ...r,
        __taxonomy: taxonomy,
        __confidence: confidence
      };
    });
  }, [data]);

  if (!normalized.length)
    return (
      <div className="bg-gradient-to-br from-white to-blue-50/40 p-6 rounded-2xl shadow-lg border border-blue-100/50">
        <div className="text-center py-10">
          <div className="text-blue-500 text-4xl mb-4">📊</div>
            <h3 className="text-lg font-medium text-blue-800 mb-2">
              No Taxonomic Data Available
            </h3>
          <p className="text-blue-600">
            Upload your eDNA samples to see taxonomic analysis
          </p>
        </div>
      </div>
    );

  // --- BUILD STATS FROM NORMALIZED DATA ---
  const taxonomyStats = normalized.reduce((acc, row) => {
    const key = row.__taxonomy || "Unknown";
    if (!acc[key]) acc[key] = { count: 0, totalConfidence: 0 };
    acc[key].count += 1;
    acc[key].totalConfidence += row.__confidence || 0;
    return acc;
  }, {});

  // Search filter
  const filteredTaxonomyStats = Object.entries(taxonomyStats).filter(
    ([taxonomy]) => taxonomy.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Rows w/ sorting
  const rows = filteredTaxonomyStats
    .map(([taxonomy, stats]) => ({
      taxonomy,
      count: stats.count,
      avgConfidence: stats.count
        ? stats.totalConfidence / stats.count
        : 0,
      percentage: (stats.count / normalized.length) * 100
    }))
    .sort((a, b) => {
      const mult = sortDirection === "asc" ? 1 : -1;
      if (sortField === "taxonomy")
        return mult * a.taxonomy.localeCompare(b.taxonomy);
      return mult * (a[sortField] - b[sortField]);
    });

  // Reset page if data length changes
  useEffect(() => {
    if (page * PAGE_SIZE >= rows.length && page !== 0) {
      setPage(0);
    }
  }, [rows.length, page]);

  const totalPages = Math.ceil(rows.length / PAGE_SIZE) || 1;
  const startIndex = page * PAGE_SIZE;
  const endIndex = Math.min(startIndex + PAGE_SIZE, rows.length);
  const paginatedRows = rows.slice(startIndex, endIndex);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(d => d === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
      setPage(0);
    }
  };

  const getConfidenceColorClass = (confidence) => {
    if (confidence >= 80) return "bg-emerald-100 text-emerald-800";
    if (confidence >= 50) return "bg-blue-100 text-blue-800";
    return "bg-amber-100 text-amber-800";
  };

  const getCountBadgeStyle = (count, index) => {
    const maxCount = Math.max(...rows.map(r => r.count));
    if (count === maxCount) return "bg-blue-600 text-white";
    if (index < 3) return "bg-blue-500 text-white";
    if (index < 5) return "bg-blue-400 text-white";
    return "bg-blue-50 text-blue-700";
  };

  // CSV full dataset (not just current page)
  const csvData = rows.map(row => ({
    Taxonomy: row.taxonomy,
    Count: row.count,
    "Average Confidence": `${row.avgConfidence.toFixed(1)}%`,
    Percentage: `${row.percentage.toFixed(1)}%`
  }));

  const totalTaxa = rows.length;
  const highConfidenceTaxa = rows.filter(r => r.avgConfidence >= 80).length;

  return (
    <div
      className={`bg-gradient-to-br from-white to-blue-50/40 p-6 rounded-2xl shadow-lg border border-blue-100/50 transition-all duration-500 ${
        isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
      }`}
    >
      <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold bg-gradient-to-r from-blue-700 to-blue-500 bg-clip-text text-transparent">
            Taxonomy Breakdown
          </h2>
          <p className="text-blue-600 text-sm mt-1">
            Analysis of {normalized.length} eDNA sequences across {totalTaxa} taxa
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="text-xs bg-blue-50 px-3 py-1 rounded-full border border-blue-200">
            <span className="text-blue-600">{rows.length}</span>
            <span className="text-blue-500"> taxonomies</span>
          </div>

          <CSVLink
            data={csvData}
            filename="taxonomy-breakdown.csv"
            className="text-xs bg-blue-600 text-white px-3 py-1 rounded-full hover:bg-blue-700 transition-colors flex items-center gap-1 shadow-sm hover:shadow"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-3 w-3"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
            Export
          </CSVLink>
        </div>
      </div>

      {/* Summary + Search */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
        <div className="bg-white/70 rounded-lg p-3 border border-blue-100 shadow-sm flex items-center space-x-3">
          <div className="bg-blue-100 p-2 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"/>
            </svg>
          </div>
          <div>
            <div className="text-xs text-blue-500">Most Abundant</div>
            <div className="font-medium text-blue-900 truncate max-w-[200px]">
              {rows.length ? rows[0].taxonomy : "N/A"}
            </div>
          </div>
        </div>
        <div className="bg-white/70 rounded-lg p-3 border border-blue-100 shadow-sm flex items-center space-x-3">
          <div className="bg-green-100 p-2 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0114 0z"/>
            </svg>
          </div>
          <div>
            <div className="text-xs text-green-500">High Confidence Taxa</div>
            <div className="font-medium text-green-900">
              {highConfidenceTaxa}{" "}
              <span className="text-xs text-green-600">
                ({((highConfidenceTaxa / (rows.length || 1)) * 100).toFixed(0)}%)
              </span>
            </div>
          </div>
        </div>
        <div className="relative bg-white/70 rounded-lg p-3 border border-blue-100 shadow-sm flex">
          <input
            type="text"
            placeholder="Search taxa..."
            value={searchTerm}
            onChange={(e) => { setSearchTerm(e.target.value); setPage(0); }}
            className="w-full pl-8 pr-4 py-1 border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-400 absolute left-6 top-1/2 -translate-y-1/2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
          </svg>
        </div>
      </div>

      {/* Table */}
      {paginatedRows.length === 0 ? (
        <div className="bg-white/70 p-10 rounded-xl text-center shadow-sm border border-blue-100">
          <div className="text-blue-400 text-2xl mb-3">🔍</div>
          <p className="text-blue-800 font-medium">No results match your search</p>
          <p className="text-blue-600 text-sm mt-1">Try a different search term</p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl shadow-sm border border-blue-100">
          <table className="w-full">
            <thead>
              <tr className="bg-gradient-to-r from-blue-600 to-blue-500 text-white">
                <th className="px-4 py-3 text-left cursor-pointer hover:bg-blue-700/50 transition-colors"
                    onClick={() => handleSort("taxonomy")}>
                  <div className="flex items-center gap-1">
                    <span>Taxonomy</span>
                    {sortField === "taxonomy" && <span>{sortDirection === "asc" ? "▲" : "▼"}</span>}
                  </div>
                </th>
                <th className="px-4 py-3 text-left cursor-pointer hover:bg-blue-700/50 transition-colors"
                    onClick={() => handleSort("count")}>
                  <div className="flex items-center gap-1">
                    <span>Count</span>
                    {sortField === "count" && <span>{sortDirection === "asc" ? "▲" : "▼"}</span>}
                  </div>
                </th>
                <th className="px-4 py-3 text-left cursor-pointer hover:bg-blue-700/50 transition-colors"
                    onClick={() => handleSort("avgConfidence")}>
                  <div className="flex items-center gap-1">
                    <span>Avg Confidence</span>
                    {sortField === "avgConfidence" && <span>{sortDirection === "asc" ? "▲" : "▼"}</span>}
                  </div>
                </th>
                <th className="px-4 py-3 text-left cursor-pointer hover:bg-blue-700/50 transition-colors"
                    onClick={() => handleSort("percentage")}>
                  <div className="flex items-center gap-1">
                    <span>Percentage</span>
                    {sortField === "percentage" && <span>{sortDirection === "asc" ? "▲" : "▼"}</span>}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-blue-100">
              {paginatedRows.map((row, i) => {
                const globalIndex = startIndex + i;
                return (
                  <tr
                    key={globalIndex}
                    className={`hover:bg-blue-50/70 transition-all cursor-pointer ${
                      activeRow === globalIndex ? "bg-blue-50" : ""
                    }`}
                    style={{ animationDelay: `${i * 0.05}s` }}
                    onMouseEnter={() => setActiveRow(globalIndex)}
                    onMouseLeave={() => setActiveRow(null)}
                  >
                    <td className="px-4 py-3">
                      <div className={`font-medium ${activeRow === globalIndex ? "text-blue-700" : "text-blue-800"}`}>
                        {row.taxonomy}
                        {row.taxonomy.toLowerCase().includes("novel") && (
                          <span className="ml-2 px-1.5 py-0.5 bg-amber-100 text-amber-800 rounded text-xs">
                            Novel
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div
                        className={`${getCountBadgeStyle(
                          row.count,
                          globalIndex
                        )} px-2 py-1 rounded-md inline-block font-medium text-center min-w-[40px]`}
                      >
                        {row.count}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center space-x-2">
                        <div className="w-16 bg-gray-200 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-2 rounded-full ${
                              row.avgConfidence >= 80
                                ? "bg-emerald-500"
                                : row.avgConfidence >= 50
                                ? "bg-blue-500"
                                : "bg-amber-500"
                            } transition-all duration-500`}
                            style={{ width: `${row.avgConfidence}%` }}
                          />
                        </div>
                        <span
                          className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColorClass(
                            row.avgConfidence
                          )}`}
                        >
                          {row.avgConfidence.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 relative">
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-gray-200 rounded-full h-2 overflow-hidden">
                          <div
                            className={`h-2 rounded-full ${
                              activeRow === globalIndex ? "bg-blue-600" : "bg-green-500"
                            } transition-all duration-300`}
                            style={{ width: `${row.percentage}%` }}
                          />
                        </div>
                        <span
                          className={`text-sm font-medium ${
                            activeRow === globalIndex ? "text-blue-700" : ""
                          }`}
                        >
                          {row.percentage.toFixed(1)}%
                        </span>
                      </div>
                      {activeRow === globalIndex && (
                        <span className="absolute right-2 bottom-1 text-xs text-blue-500">
                          {row.count} of {normalized.length}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mt-4 text-xs">
        <div className="text-gray-600">
          Showing {startIndex + 1}-{endIndex} of {rows.length} taxa
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            className={`px-3 py-1 rounded border text-xs font-medium ${
              page === 0
                ? "bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed"
                : "bg-white text-blue-700 border-blue-300 hover:bg-blue-50"
            }`}
          >
            Prev
          </button>
          <div className="flex items-center gap-1">
            <span className="text-gray-500">Page</span>
            <span className="text-blue-700 font-semibold">
              {page + 1}
            </span>
            <span className="text-gray-500">/ {totalPages}</span>
          </div>
          <button
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            className={`px-3 py-1 rounded border text-xs font-medium ${
              page >= totalPages - 1
                ? "bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed"
                : "bg-white text-blue-700 border-blue-300 hover:bg-blue-50"
            }`}
          >
            Next
          </button>
        </div>
      </div>

      <div className="flex justify-between items-center mt-3 text-[10px] text-gray-500">
        <span>Click column headers to sort</span>
        <span>Export includes all rows</span>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        tbody tr {
          animation: fadeIn 0.3s ease-out forwards;
        }
      `}</style>
    </div>
  );
};

export default TaxonomyTable;