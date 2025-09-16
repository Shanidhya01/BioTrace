import React, { useState, useEffect } from "react";
import { CSVLink } from "react-csv";

const TaxonomyTable = ({ data }) => {
  const [sortField, setSortField] = useState("count");
  const [sortDirection, setSortDirection] = useState("desc");
  const [searchTerm, setSearchTerm] = useState("");
  const [activeRow, setActiveRow] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  if (!data || data.length === 0) return (
    <div className="bg-gradient-to-br from-white to-blue-50/40 p-6 rounded-2xl shadow-lg border border-blue-100/50">
      <div className="text-center py-10">
        <div className="text-blue-500 text-4xl mb-4">📊</div>
        <h3 className="text-lg font-medium text-blue-800 mb-2">No Taxonomic Data Available</h3>
        <p className="text-blue-600">Upload your eDNA samples to see taxonomic analysis</p>
      </div>
    </div>
  );

  // Count by taxonomy with confidence average
  const taxonomyStats = data.reduce((acc, row) => {
    const key = row.Taxonomy || "Unknown";
    if (!acc[key]) {
      acc[key] = { count: 0, totalConfidence: 0 };
    }
    acc[key].count += 1;
    acc[key].totalConfidence += row.Confidence || 0;
    return acc;
  }, {});

  // Apply search filter
  const filteredTaxonomyStats = Object.entries(taxonomyStats)
    .filter(([taxonomy]) => 
      taxonomy.toLowerCase().includes(searchTerm.toLowerCase())
    );

  // Create and sort rows based on current sort settings
  const rows = filteredTaxonomyStats
    .map(([taxonomy, stats]) => ({
      taxonomy,
      count: stats.count,
      avgConfidence: (stats.totalConfidence / stats.count) * 100,
      percentage: (stats.count / data.length) * 100
    }))
    .sort((a, b) => {
      const multiplier = sortDirection === "asc" ? 1 : -1;
      if (sortField === "taxonomy") {
        return multiplier * a.taxonomy.localeCompare(b.taxonomy);
      }
      return multiplier * (a[sortField] - b[sortField]);
    });

  // Handle column sort
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  // Get color class based on confidence value
  const getConfidenceColorClass = (confidence) => {
    if (confidence >= 80) return "bg-emerald-100 text-emerald-800";
    if (confidence >= 50) return "bg-blue-100 text-blue-800";
    return "bg-amber-100 text-amber-800";
  };

  // Get badge style based on count ranking
  const getCountBadgeStyle = (count, index) => {
    const maxCount = Math.max(...rows.map(r => r.count));
    
    if (count === maxCount) return "bg-blue-600 text-white";
    if (index < 3) return "bg-blue-500 text-white";
    if (index < 5) return "bg-blue-400 text-white";
    return "bg-blue-50 text-blue-700";
  };

  // Prepare CSV data for export
  const csvData = rows.map(row => ({
    Taxonomy: row.taxonomy,
    Count: row.count,
    'Average Confidence': `${row.avgConfidence.toFixed(1)}%`,
    Percentage: `${row.percentage.toFixed(1)}%`
  }));

  // Calculate summary stats
  const totalTaxa = rows.length;
  const highConfidenceTaxa = rows.filter(row => row.avgConfidence >= 80).length;
  
  return (
    <div className={`bg-gradient-to-br from-white to-blue-50/40 p-6 rounded-2xl shadow-lg border border-blue-100/50 transition-all duration-500 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
      <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold bg-gradient-to-r from-blue-700 to-blue-500 bg-clip-text text-transparent">
            Taxonomy Breakdown
          </h2>
          <p className="text-blue-600 text-sm mt-1">
            Analysis of {data.length} eDNA sequences across {totalTaxa} taxa
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
            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
            Export
          </CSVLink>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
        <div className="bg-white/70 rounded-lg p-3 border border-blue-100 shadow-sm flex items-center space-x-3">
          <div className="bg-blue-100 p-2 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
            </svg>
          </div>
          <div>
            <div className="text-xs text-blue-500">Most Abundant</div>
            <div className="font-medium text-blue-900 truncate max-w-[200px]">
              {rows.length > 0 ? rows[0].taxonomy : "N/A"}
            </div>
          </div>
        </div>
        
        <div className="bg-white/70 rounded-lg p-3 border border-blue-100 shadow-sm flex items-center space-x-3">
          <div className="bg-green-100 p-2 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0114 0z" />
            </svg>
          </div>
          <div>
            <div className="text-xs text-green-500">High Confidence Taxa</div>
            <div className="font-medium text-green-900">
              {highConfidenceTaxa} <span className="text-xs text-green-600">({((highConfidenceTaxa/totalTaxa)*100).toFixed(0)}%)</span>
            </div>
          </div>
        </div>
        
        <div className="relative bg-white/70 rounded-lg p-3 border border-blue-100 shadow-sm flex">
          <input
            type="text"
            placeholder="Search taxa..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-4 py-1 border border-blue-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-blue-400 absolute left-6 top-[50%] transform -translate-y-1/2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>
      
      {rows.length === 0 ? (
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
                <th 
                  className="px-4 py-3 text-left cursor-pointer hover:bg-blue-700/50 transition-colors" 
                  onClick={() => handleSort("taxonomy")}
                >
                  <div className="flex items-center gap-1">
                    <span>Taxonomy</span>
                    {sortField === "taxonomy" && (
                      <span>{sortDirection === "asc" ? "▲" : "▼"}</span>
                    )}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 text-left cursor-pointer hover:bg-blue-700/50 transition-colors" 
                  onClick={() => handleSort("count")}
                >
                  <div className="flex items-center gap-1">
                    <span>Count</span>
                    {sortField === "count" && (
                      <span>{sortDirection === "asc" ? "▲" : "▼"}</span>
                    )}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 text-left cursor-pointer hover:bg-blue-700/50 transition-colors" 
                  onClick={() => handleSort("avgConfidence")}
                >
                  <div className="flex items-center gap-1">
                    <span>Avg Confidence</span>
                    {sortField === "avgConfidence" && (
                      <span>{sortDirection === "asc" ? "▲" : "▼"}</span>
                    )}
                  </div>
                </th>
                <th 
                  className="px-4 py-3 text-left cursor-pointer hover:bg-blue-700/50 transition-colors" 
                  onClick={() => handleSort("percentage")}
                >
                  <div className="flex items-center gap-1">
                    <span>Percentage</span>
                    {sortField === "percentage" && (
                      <span>{sortDirection === "asc" ? "▲" : "▼"}</span>
                    )}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-blue-100">
              {rows.map((row, i) => (
                <tr 
                  key={i} 
                  className={`hover:bg-blue-50/70 transition-all cursor-pointer ${activeRow === i ? 'bg-blue-50' : ''}`}
                  style={{ animationDelay: `${i * 0.05}s` }}
                  onMouseEnter={() => setActiveRow(i)}
                  onMouseLeave={() => setActiveRow(null)}
                >
                  <td className="px-4 py-3">
                    <div className={`font-medium ${activeRow === i ? 'text-blue-700' : 'text-blue-800'}`}>
                      {row.taxonomy}
                      {row.taxonomy.toLowerCase().includes("novel") && (
                        <span className="ml-2 px-1.5 py-0.5 bg-amber-100 text-amber-800 rounded text-xs">Novel</span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className={`${getCountBadgeStyle(row.count, i)} px-2 py-1 rounded-md inline-block font-medium text-center min-w-[40px]`}>
                      {row.count}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center space-x-2">
                      <div className="w-16 bg-gray-200 rounded-full h-2 overflow-hidden">
                        <div 
                          className={`h-2 rounded-full ${row.avgConfidence >= 80 ? 'bg-emerald-500' : row.avgConfidence >= 50 ? 'bg-blue-500' : 'bg-amber-500'} transition-all duration-500`}
                          style={{width: `${row.avgConfidence}%`}}
                        ></div>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColorClass(row.avgConfidence)}`}>
                        {row.avgConfidence.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 relative">
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2 overflow-hidden">
                        <div 
                          className={`h-2 rounded-full ${activeRow === i ? 'bg-blue-600' : 'bg-green-500'} transition-all duration-300`}
                          style={{width: `${row.percentage}%`}}
                        ></div>
                      </div>
                      <span className={`text-sm font-medium ${activeRow === i ? 'text-blue-700' : ''}`}>
                        {row.percentage.toFixed(1)}%
                      </span>
                    </div>
                    {activeRow === i && (
                      <span className="absolute right-2 bottom-1 text-xs text-blue-500">
                        {row.count} of {data.length}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      
      <div className="flex justify-between items-center mt-4 text-xs text-gray-500">
        <div>Showing {rows.length} of {Object.keys(taxonomyStats).length} taxonomies</div>
        <div>Click column headers to sort</div>
      </div>
      
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        tr {
          animation: fadeIn 0.3s ease-out forwards;
        }
      `}</style>
    </div>
  );
};

export default TaxonomyTable;