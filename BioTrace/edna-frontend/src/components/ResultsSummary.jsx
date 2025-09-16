import React from 'react';

const ResultsSummary = ({ data }) => {
  // Guard clause if no data is available
  if (!data || !data.results || !data.results.length) {
    return (
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-blue-800 mb-2">Analysis Summary</h2>
        <p className="text-blue-600">No analysis results available yet.</p>
      </div>
    );
  }

  const { results, alpha_diversity } = data;
  
  // Calculate summary statistics
  const totalSamples = results.length;
  
  // Get unique species count (excluding Novel duplicates)
  const uniqueSpecies = new Set(
    results.map(item => item.predicted_species.startsWith("Novel_") 
      ? item.cluster_id || item.predicted_species 
      : item.predicted_species)
  ).size;
  
  // Get count of high confidence predictions
  const highConfidenceResults = results.filter(item => item.confidence > 0.7).length;
  
  // Calculate status distribution
  const statusCounts = results.reduce((acc, item) => {
    acc[item.status] = (acc[item.status] || 0) + 1;
    return acc;
  }, {});
  
  // Calculate most common species in top predictions
  const topPredictionSpecies = {};
  results.forEach(item => {
    item.top_predictions.forEach(pred => {
      topPredictionSpecies[pred.species] = (topPredictionSpecies[pred.species] || 0) + 1;
    });
  });
  
  const mostCommonSpecies = Object.entries(topPredictionSpecies)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  return (
    <div className="mb-6 text-blue-900">
      <h2 className="text-xl font-semibold mb-4">eDNA Analysis Summary</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50/80 p-4 rounded-lg shadow-inner border border-blue-200/50">
          <h3 className="text-sm font-medium text-blue-700">Total Sequences</h3>
          <p className="text-2xl font-bold">{totalSamples}</p>
        </div>
        
        <div className="bg-green-50/80 p-4 rounded-lg shadow-inner border border-green-200/50">
          <h3 className="text-sm font-medium text-green-700">Species Richness</h3>
          <p className="text-2xl font-bold">{alpha_diversity?.richness || uniqueSpecies}</p>
        </div>
        
        <div className="bg-purple-50/80 p-4 rounded-lg shadow-inner border border-purple-200/50">
          <h3 className="text-sm font-medium text-purple-700">Shannon Diversity</h3>
          <p className="text-2xl font-bold">{alpha_diversity?.shannon_index?.toFixed(2) || "N/A"}</p>
          <p className="text-xs text-purple-600">Evenness: {alpha_diversity?.evenness?.toFixed(2) || "N/A"}</p>
        </div>
        
        <div className="bg-amber-50/80 p-4 rounded-lg shadow-inner border border-amber-200/50">
          <h3 className="text-sm font-medium text-amber-700">High Confidence</h3>
          <p className="text-2xl font-bold">{highConfidenceResults}</p>
          <p className="text-xs text-amber-600">({((highConfidenceResults/totalSamples)*100).toFixed(1)}% of total)</p>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Status breakdown */}
        <div className="bg-white/50 rounded-lg p-4 border border-blue-100/50">
          <h3 className="text-md font-semibold mb-3">Sequence Classification</h3>
          <div className="space-y-2">
            {Object.entries(statusCounts).map(([status, count]) => (
              <div key={status} className="flex items-center justify-between">
                <span className="font-medium capitalize">{status.replace('_', ' ')}</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-blue-100/50 rounded-full h-2.5">
                    <div 
                      className={`h-2.5 rounded-full ${status === 'novel' ? 'bg-amber-500' : 'bg-emerald-500'}`}
                      style={{width: `${(count/totalSamples)*100}%`}}
                    ></div>
                  </div>
                  <span className="text-sm">{count} ({((count/totalSamples)*100).toFixed(1)}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Most common species */}
        <div className="bg-white/50 rounded-lg p-4 border border-blue-100/50">
          <h3 className="text-md font-semibold mb-3">Most Common Species</h3>
          <div className="space-y-2">
            {mostCommonSpecies.map(([species, count], index) => (
              <div key={species} className="flex items-center justify-between">
                <span className="font-medium truncate max-w-[180px]">{species}</span>
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                  {count} references ({((count/results.length)*100).toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsSummary;