import React, { useState, useMemo, useEffect, useRef } from "react";
import { 
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, LineChart, Line, AreaChart, Area, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Treemap, ComposedChart, Scatter, ScatterChart, ZAxis,
  Brush, ReferenceLine
} from 'recharts';
import html2canvas from 'html2canvas'; // npm install html2canvas

const Charts = ({ data }) => {
  const [chartType, setChartType] = useState('default');
  const [showCount, setShowCount] = useState(10);
  const [theme, setTheme] = useState('ocean'); // 'ocean', 'forest', 'sunset'
  const [isAnimating, setIsAnimating] = useState(false);
  const [showInsights, setShowInsights] = useState(false);
  const chartRef = useRef(null);
  
  // Handle animation on chart type change
  useEffect(() => {
    setIsAnimating(true);
    const timer = setTimeout(() => setIsAnimating(false), 300);
    return () => clearTimeout(timer);
  }, [chartType]);

  if (!data || data.length === 0) return (
    <div className="bg-gradient-to-br from-white to-blue-50/40 p-6 rounded-2xl shadow-lg border border-blue-100/50">
      <h2 className="text-xl font-bold bg-gradient-to-r from-blue-700 to-blue-500 bg-clip-text text-transparent mb-4">Data Visualization</h2>
      <div className="text-center py-10">
        <div className="text-blue-500 text-4xl mb-4">📊</div>
        <h3 className="text-lg font-medium text-blue-800 mb-2">No Visualization Data Available</h3>
        <p className="text-blue-600">Upload your eDNA samples to see visualizations</p>
      </div>
    </div>
  );

  // Get theme colors
  const getThemeColors = () => {
    switch(theme) {
      case 'forest':
        return [
          '#047857', '#10b981', '#34d399', '#6ee7b7', '#a7f3d0',
          '#064e3b', '#065f46', '#059669', '#0d9488', '#14b8a6'
        ];
      case 'sunset':
        return [
          '#b91c1c', '#dc2626', '#ef4444', '#f87171', '#fca5a5',
          '#7c2d12', '#9a3412', '#c2410c', '#ea580c', '#f97316'
        ];
      default: // ocean
        return [
          '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe',
          '#1d4ed8', '#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd',
          '#0369a1', '#0284c7', '#0ea5e9', '#38bdf8', '#7dd3fc'
        ];
    }
  };
  
  const COLORS = getThemeColors();

  // Prepare data for charts
  const taxonomyCounts = data.reduce((acc, row) => {
    const key = row.Taxonomy || "Unknown";
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  // Calculate confidence data with percentages
  const confidenceData = useMemo(() => {
    const ranges = {
      '90-100%': 0, 
      '80-90%': 0,
      '70-80%': 0,
      '60-70%': 0,
      '50-60%': 0,
      '<50%': 0
    };
    
    data.forEach(item => {
      const confidence = (item.Confidence || 0) * 100;
      if (confidence >= 90) ranges['90-100%']++;
      else if (confidence >= 80) ranges['80-90%']++;
      else if (confidence >= 70) ranges['70-80%']++;
      else if (confidence >= 60) ranges['60-70%']++;
      else if (confidence >= 50) ranges['50-60%']++;
      else ranges['<50%']++;
    });
    
    return Object.entries(ranges).map(([name, value]) => ({ 
      name, 
      value, 
      percentage: (value / data.length * 100).toFixed(1)
    }));
  }, [data]);

  // Get top species data
  const chartData = useMemo(() => {
    return Object.entries(taxonomyCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, showCount)
      .map(([name, count], index) => ({
        name: name.length > 15 ? name.substring(0, 15) + '...' : name,
        count,
        fullName: name,
        rank: index + 1,
        confidence: data.filter(row => row.Taxonomy === name).reduce((sum, row) => sum + (row.Confidence || 0), 0) / count
      }));
  }, [taxonomyCounts, showCount, data]);

  // Pie chart data
  const pieData = useMemo(() => {
    const totalOther = Object.values(taxonomyCounts).reduce((a, b) => a + b, 0) - 
                      chartData.reduce((a, b) => a + b.count, 0);
    
    return [
      ...chartData,
      ...(totalOther > 0 ? [{ name: 'Other', count: totalOther, fullName: 'Other Species' }] : [])
    ];
  }, [taxonomyCounts, chartData]);

  // Scatter data for confidence vs count
  const scatterData = useMemo(() => {
    return Object.entries(taxonomyCounts)
      .map(([name, count]) => {
        const avgConfidence = data.filter(row => row.Taxonomy === name)
          .reduce((sum, row) => sum + (row.Confidence || 0), 0) / count;
        return {
          name: name.length > 15 ? name.substring(0, 15) + '...' : name,
          fullName: name,
          count,
          confidence: avgConfidence * 100,
          size: Math.min(count * 2, 30)  // Size proportional to count, but capped
        };
      });
  }, [taxonomyCounts, data]);

  // Calculate novel vs known
  const novelVsKnown = useMemo(() => {
    const novelCount = data.filter(item => 
      item.Taxonomy?.toLowerCase().includes('novel') || 
      item.status === 'novel'
    ).length;

    return [
      { name: 'Novel', value: novelCount },
      { name: 'Known', value: data.length - novelCount }
    ];
  }, [data]);

  // Trend data (simulation based on confidence)
  const trendData = useMemo(() => {
    // Group by confidence ranges for trend simulation
    const confidenceRanges = [0.9, 0.8, 0.7, 0.6, 0.5, 0];
    const result = confidenceRanges.map((minConf, idx) => {
      const maxConf = idx > 0 ? confidenceRanges[idx-1] : 1;
      const count = data.filter(row => 
        (row.Confidence || 0) >= minConf && (row.Confidence || 0) < maxConf
      ).length;
      
      // Add some simulated time points
      return {
        name: minConf === 0 ? '<50%' : `${minConf*100}-${maxConf*100}%`,
        week1: Math.round(count * 0.4),
        week2: Math.round(count * 0.7),
        week3: count,
        avg: (count * 0.4 + count * 0.7 + count) / 3
      };
    });
    
    return result;
  }, [data]);

  // Generate insights
  const insights = useMemo(() => {
    if (!data.length) return [];
    
    const results = [];
    
    // Most common species
    if (chartData.length > 0) {
      results.push({
        icon: "🔍",
        color: "text-blue-700",
        text: `Most common species is "${chartData[0].fullName}" with ${chartData[0].count} sequences (${((chartData[0].count/data.length)*100).toFixed(1)}% of total).`
      });
    }
    
    // Confidence distribution
    const highConfidence = confidenceData.find(d => d.name === '90-100%')?.value || 0;
    const lowConfidence = confidenceData.find(d => d.name === '<50%')?.value || 0;
    
    if (highConfidence > lowConfidence && highConfidence > data.length * 0.3) {
      results.push({
        icon: "✅",
        color: "text-green-700", 
        text: `Strong confidence in species identification with ${highConfidence} sequences (${((highConfidence/data.length)*100).toFixed(1)}%) having >90% confidence.`
      });
    } else if (lowConfidence > data.length * 0.3) {
      results.push({
        icon: "⚠️",
        color: "text-amber-700",
        text: `Limited identification confidence with ${lowConfidence} sequences (${((lowConfidence/data.length)*100).toFixed(1)}%) having <50% confidence.`
      });
    }
    
    // Biodiversity
    const speciesCount = Object.keys(taxonomyCounts).length;
    if (speciesCount > 10) {
      results.push({
        icon: "🌿",
        color: "text-emerald-700",
        text: `High biodiversity detected with ${speciesCount} different species across ${data.length} samples.`
      });
    }
    
    // Novel species
    const novelCount = novelVsKnown.find(d => d.name === 'Novel')?.value || 0;
    if (novelCount > 0) {
      results.push({
        icon: "🔬",
        color: "text-purple-700",
        text: `${novelCount} potential novel species detected (${((novelCount/data.length)*100).toFixed(1)}% of sequences).`
      });
    }
    
    return results;
  }, [data, chartData, confidenceData, taxonomyCounts, novelVsKnown]);

  // Export chart as image
  const exportChart = async () => {
    if (chartRef.current) {
      const canvas = await html2canvas(chartRef.current);
      const image = canvas.toDataURL("image/png");
      const link = document.createElement('a');
      link.href = image;
      link.download = `edna-chart-${chartType}-${new Date().toISOString().slice(0,10)}.png`;
      link.click();
    }
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-blue-100">
          <p className="font-medium text-blue-800">{data.fullName || label}</p>
          <p className="text-blue-600">
            <span className="font-medium">{payload[0].value}</span> {payload[0].name === "value" ? "sequences" : "records"}
          </p>
          {data.confidence !== undefined && (
            <p className="text-xs text-green-600 mt-1">
              Avg confidence: {(data.confidence * 100).toFixed(1)}%
            </p>
          )}
          <p className="text-xs text-blue-500 mt-1">
            {((payload[0].value / data.length) * 100).toFixed(1)}% of total
          </p>
        </div>
      );
    }
    return null;
  };

  const CustomScatterTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-blue-100">
          <p className="font-medium text-blue-800">{payload[0].payload.fullName}</p>
          <p className="text-blue-600">Count: <span className="font-medium">{payload[0].value}</span></p>
          <p className="text-green-600">Confidence: <span className="font-medium">{payload[1].value.toFixed(1)}%</span></p>
        </div>
      );
    }
    return null;
  };

  const renderChart = () => {
    switch (chartType) {
      case 'radar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart outerRadius={150} data={chartData.slice(0, 8)}>
              <PolarGrid strokeDasharray="3 3" stroke="#bfdbfe" />
              <PolarAngleAxis dataKey="name" tick={{ fill: '#3b82f6', fontSize: 12 }} />
              <PolarRadiusAxis stroke="#60a5fa" />
              <Radar name="Species Count" dataKey="count" stroke={COLORS[0]} 
                fill={COLORS[0]} fillOpacity={0.6} animationDuration={1500} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        );
      
      case 'area':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS[0]} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={COLORS[0]} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#bfdbfe" />
              <XAxis dataKey="name" tick={{ fill: '#1e40af', fontSize: 12 }} />
              <YAxis tick={{ fill: '#1e40af' }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="count" stroke={COLORS[0]} 
                    fillOpacity={1} fill="url(#colorCount)" animationDuration={1500} />
            </AreaChart>
          </ResponsiveContainer>
        );
      
      case 'confidence':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={confidenceData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#bfdbfe" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#1e40af' }} />
              <YAxis dataKey="name" type="category" tick={{ fill: '#1e40af', fontSize: 12 }} width={80} />
              <Tooltip />
              <Legend />
              <Bar dataKey="value" name="Sequence Count" animationDuration={1500}>
                {confidenceData.map((entry, index) => {
                  // Color based on confidence level
                  let color;
                  if (entry.name === '90-100%' || entry.name === '80-90%') color = theme === 'forest' ? '#047857' : theme === 'sunset' ? '#b91c1c' : '#059669';
                  else if (entry.name === '70-80%' || entry.name === '60-70%') color = theme === 'forest' ? '#10b981' : theme === 'sunset' ? '#dc2626' : '#2563eb';
                  else color = theme === 'forest' ? '#34d399' : theme === 'sunset' ? '#f87171' : '#d97706';
                  
                  return <Cell key={`cell-${index}`} fill={color} />
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        );
        
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <defs>
                {COLORS.map((color, index) => (
                  <linearGradient key={`gradient-${index}`} id={`gradient-${index}`} x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity={1} />
                    <stop offset="100%" stopColor={color} stopOpacity={0.7} />
                  </linearGradient>
                ))}
              </defs>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={70}
                outerRadius={120}
                fill="#8884d8"
                paddingAngle={1}
                dataKey="count"
                animationDuration={1500}
                animationBegin={0}
                label={({ name, percent }) => percent > 0.05 ? `${name} (${(percent * 100).toFixed(0)}%)` : ''}
                labelLine={false}
              >
                {pieData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={`url(#gradient-${index % COLORS.length})`} 
                    stroke="#fff"
                    strokeWidth={1}
                  />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend 
                layout="vertical"
                verticalAlign="middle"
                align="right"
                wrapperStyle={{ fontSize: '12px' }}
              />
            </PieChart>
          </ResponsiveContainer>
        );
      
      case 'treemap':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <Treemap
              data={chartData}
              dataKey="count"
              aspectRatio={4/3}
              stroke="#fff"
              fill={COLORS[0]}
              animationDuration={1500}
            >
              {chartData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
              <Tooltip content={<CustomTooltip />} />
            </Treemap>
          </ResponsiveContainer>
        );

      case 'scatter':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#bfdbfe" />
              <XAxis 
                type="number" 
                dataKey="count" 
                name="Count" 
                tick={{ fill: '#1e40af' }}
                label={{ value: 'Sequence Count', position: 'insideBottom', offset: -5, fill: '#1e40af' }}
              />
              <YAxis 
                type="number" 
                dataKey="confidence" 
                name="Confidence" 
                tick={{ fill: '#1e40af' }}
                label={{ value: 'Confidence %', angle: -90, position: 'insideLeft', fill: '#1e40af' }}
              />
              <ZAxis dataKey="size" range={[20, 500]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomScatterTooltip />} />
              <Scatter name="Species" data={scatterData} fill={COLORS[0]}>
                {scatterData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={COLORS[index % COLORS.length]} 
                  />
                ))}
              </Scatter>
              <ReferenceLine y={80} stroke="green" strokeDasharray="3 3" label={{ value: "High confidence", position: "insideTopRight", fill: "green" }} />
              <ReferenceLine y={50} stroke="orange" strokeDasharray="3 3" label={{ value: "Medium confidence", position: "insideTopRight", fill: "orange" }} />
            </ScatterChart>
          </ResponsiveContainer>
        );

      case 'novelty':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <defs>
                <linearGradient id="novelGradient" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor={theme === 'forest' ? '#047857' : theme === 'sunset' ? '#b91c1c' : '#7c3aed'} stopOpacity={1} />
                  <stop offset="100%" stopColor={theme === 'forest' ? '#10b981' : theme === 'sunset' ? '#dc2626' : '#8b5cf6'} stopOpacity={0.7} />
                </linearGradient>
                <linearGradient id="knownGradient" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor={theme === 'forest' ? '#0d9488' : theme === 'sunset' ? '#ea580c' : '#2563eb'} stopOpacity={1} />
                  <stop offset="100%" stopColor={theme === 'forest' ? '#14b8a6' : theme === 'sunset' ? '#f97316' : '#3b82f6'} stopOpacity={0.7} />
                </linearGradient>
              </defs>
              <Pie
                data={novelVsKnown}
                cx="50%"
                cy="50%"
                innerRadius={80}
                outerRadius={140}
                fill="#8884d8"
                paddingAngle={2}
                dataKey="value"
                animationDuration={1500}
                animationBegin={0}
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                labelLine={true}
              >
                <Cell fill="url(#novelGradient)" stroke="#fff" strokeWidth={2} />
                <Cell fill="url(#knownGradient)" stroke="#fff" strokeWidth={2} />
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );

      case 'trend':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={trendData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid stroke="#bfdbfe" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fill: '#1e40af' }} />
              <YAxis tick={{ fill: '#1e40af' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="week1" barSize={20} fill={COLORS[0]} name="Week 1" />
              <Bar dataKey="week2" barSize={20} fill={COLORS[1]} name="Week 2" />
              <Bar dataKey="week3" barSize={20} fill={COLORS[2]} name="Week 3" />
              <Line type="monotone" dataKey="avg" stroke={theme === 'forest' ? '#047857' : theme === 'sunset' ? '#b91c1c' : '#7c3aed'} name="Average" />
            </ComposedChart>
          </ResponsiveContainer>
        );
        
      default:
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 70 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#bfdbfe" />
              <XAxis 
                dataKey="name" 
                tick={{ fill: '#1e40af', fontSize: 12 }}
                interval={0}
                angle={-45}
                textAnchor="end"
              />
              <YAxis tick={{ fill: '#1e40af' }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ paddingTop: 15 }} />
              <Bar 
                name="Sequence Count" 
                dataKey="count" 
                animationDuration={1500}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
              <Brush dataKey="name" height={20} stroke={COLORS[0]} />
            </BarChart>
          </ResponsiveContainer>
        );
    }
  };

  return (
    <div className={`bg-gradient-to-br from-white to-blue-50/40 p-6 rounded-2xl shadow-lg border border-blue-100/50 transition-all duration-300 hover:shadow-xl ${theme === 'forest' ? 'from-white to-green-50/40 border-green-100/50' : theme === 'sunset' ? 'from-white to-orange-50/40 border-orange-100/50' : ''}`}>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <h2 className={`text-xl font-bold bg-gradient-to-r bg-clip-text text-transparent ${theme === 'forest' ? 'from-green-700 to-green-500' : theme === 'sunset' ? 'from-red-700 to-orange-500' : 'from-blue-700 to-blue-500'}`}>
          eDNA Data Visualization
        </h2>
        
        <div className="flex flex-wrap gap-2">
          <select
            value={showCount}
            onChange={(e) => setShowCount(Number(e.target.value))}
            className={`text-sm px-3 py-1 border rounded-lg focus:outline-none focus:ring-2 bg-white/80 ${theme === 'forest' ? 'border-green-200 focus:ring-green-500' : theme === 'sunset' ? 'border-orange-200 focus:ring-orange-500' : 'border-blue-200 focus:ring-blue-500'}`}
          >
            <option value={5}>Top 5</option>
            <option value={10}>Top 10</option>
            <option value={15}>Top 15</option>
            <option value={20}>Top 20</option>
          </select>

          <select
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            className={`text-sm px-3 py-1 border rounded-lg focus:outline-none focus:ring-2 bg-white/80 ${theme === 'forest' ? 'border-green-200 focus:ring-green-500' : theme === 'sunset' ? 'border-orange-200 focus:ring-orange-500' : 'border-blue-200 focus:ring-blue-500'}`}
          >
            <option value="ocean">Ocean Theme</option>
            <option value="forest">Forest Theme</option>
            <option value="sunset">Sunset Theme</option>
          </select>
          
          <button 
            onClick={exportChart}
            className={`text-xs px-3 py-1.5 rounded-lg flex items-center gap-1 text-white ${theme === 'forest' ? 'bg-green-600 hover:bg-green-700' : theme === 'sunset' ? 'bg-orange-600 hover:bg-orange-700' : 'bg-blue-600 hover:bg-blue-700'} transition-colors`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
              <path d="M5.5 13.5a.5.5 0 0 1-.5-.5v-2a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2zm4 0a.5.5 0 0 1-.5-.5v-2a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2zm4 0a.5.5 0 0 1-.5-.5v-2a.5.5 0 0 1 .5-.5h2a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-.5.5h-2z"/>
              <path d="M.5 3a.5.5 0 0 0-.5.5v9a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-9a.5.5 0 0 0-.5-.5h-1zm16 0a.5.5 0 0 0-.5.5v9a.5.5 0 0 0 .5.5h1a.5.5 0 0 0 .5-.5v-9a.5.5 0 0 0-.5-.5h-1zm-16-1h18a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1h-18a1 1 0 0 1-1-1v-10a1 1 0 0 1 1-1zm9.5 5.5a.5.5 0 0 0-1 0v3a.5.5 0 0 0 1 0v-3z"/>
            </svg>
            Export
          </button>
        </div>
      </div>
      
      {/* Chart type selector */}
      <div className={`mb-5 flex flex-wrap gap-1 border rounded-lg overflow-hidden ${theme === 'forest' ? 'border-green-200' : theme === 'sunset' ? 'border-orange-200' : 'border-blue-200'}`}>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'default' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('default')}
          aria-label="Bar chart"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zm6-4a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zm6-3a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
          </svg>
          Bar
        </button>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'pie' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('pie')}
          aria-label="Pie chart"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z" />
            <path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z" />
          </svg>
          Pie
        </button>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'area' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('area')}
          aria-label="Area chart"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zm-2 4a5 5 0 00-4.546 2.916A5.986 5.986 0 005 10a6 6 0 0012 0c0-1.146-.322-2.217-.86-3.134A5 5 0 0010 11z" clipRule="evenodd" />
          </svg>
          Area
        </button>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'radar' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('radar')}
          aria-label="Radar chart"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
            <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
          </svg>
          Radar
        </button>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'confidence' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('confidence')}
          aria-label="Confidence distribution"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11 4a1 1 0 10-2 0v4a1 1 0 102 0V7zm-3 1a1 1 0 10-2 0v3a1 1 0 102 0V8zM8 9a1 1 0 00-2 0v2a1 1 0 102 0V9z" clipRule="evenodd" />
          </svg>
          Confidence
        </button>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'treemap' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('treemap')}
          aria-label="Treemap visualization"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM14 11a1 1 0 011 1v1h1a1 1 0 110 2h-1v1a1 1 0 11-2 0v-1h-1a1 1 0 110-2h1v-1a1 1 0 011-1z" />
          </svg>
          Treemap
        </button>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'scatter' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('scatter')}
          aria-label="Scatter plot"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" />
          </svg>
          Scatter
        </button>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'novelty' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('novelty')}
          aria-label="Novel vs known species"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
          </svg>
          Novelty
        </button>
        <button 
          className={`text-xs px-3 py-2 flex items-center gap-1 ${chartType === 'trend' ? 
            (theme === 'forest' ? 'bg-green-100 text-green-800' : 
             theme === 'sunset' ? 'bg-orange-100 text-orange-800' : 
             'bg-blue-100 text-blue-800') : 
            'bg-white hover:bg-gray-50'}`}
          onClick={() => setChartType('trend')}
          aria-label="Time trend visualization"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M3 3a1 1 0 000 2h10a1 1 0 100-2H3zM3 7a1 1 0 000 2h10a1 1 0 100-2H3zM3 11a1 1 0 100 2h10a1 1 0 100-2H3zM17 6a1 1 0 10-2 0v8a1 1 0 102 0V6z" />
          </svg>
          Trend
        </button>
      </div>

      <div className={`bg-white/70 p-4 rounded-xl border shadow-sm mb-4 transition-opacity duration-300 ${isAnimating ? 'opacity-0' : 'opacity-100'} ${theme === 'forest' ? 'border-green-100' : theme === 'sunset' ? 'border-orange-100' : 'border-blue-100'}`} ref={chartRef}>
        <div className="flex justify-between items-center mb-2">
          <h3 className={`font-semibold ${theme === 'forest' ? 'text-green-900' : theme === 'sunset' ? 'text-orange-900' : 'text-blue-900'}`}>
            {chartType === 'confidence' ? 'Confidence Distribution' : 
             chartType === 'radar' ? 'Species Radar Analysis' : 
             chartType === 'pie' ? 'Species Distribution' : 
             chartType === 'area' ? 'Species Area Distribution' :
             chartType === 'treemap' ? 'Species Proportion Treemap' :
             chartType === 'scatter' ? 'Count vs Confidence Analysis' :
             chartType === 'novelty' ? 'Novel vs Known Species' :
             chartType === 'trend' ? 'Temporal Trends (Simulated)' :
             'Top Species Breakdown'}
          </h3>
          <div className={`text-xs px-2 py-0.5 rounded-full ${theme === 'forest' ? 'bg-green-50 text-green-600' : theme === 'sunset' ? 'bg-orange-50 text-orange-600' : 'bg-blue-50 text-blue-600'}`}>
            Total: {data.length} sequences
          </div>
        </div>
        
        {renderChart()}
      </div>

      {/* Insights panel */}
      <div className={`mb-4 border-t ${theme === 'forest' ? 'border-green-100' : theme === 'sunset' ? 'border-orange-100' : 'border-blue-100'} pt-3`}>
        <button 
          className={`flex items-center gap-1 text-xs font-medium ${theme === 'forest' ? 'text-green-600' : theme === 'sunset' ? 'text-orange-600' : 'text-blue-600'} mb-2`}
          onClick={() => setShowInsights(!showInsights)}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className={`h-3.5 w-3.5 transition-transform ${showInsights ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          {showInsights ? 'Hide Insights' : 'Show AI Insights'}
        </button>
        
        {showInsights && insights.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3 animate-fadeIn">
            {insights.map((insight, idx) => (
              <div key={idx} className={`p-3 rounded-lg bg-white/60 border ${theme === 'forest' ? 'border-green-100' : theme === 'sunset' ? 'border-orange-100' : 'border-blue-100'} flex gap-3 items-start`}>
                <div className="text-lg">{insight.icon}</div>
                <div className={`text-sm ${insight.color}`}>{insight.text}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex justify-between items-center text-xs text-gray-500">
        <span>Click chart legend items to toggle visibility</span>
        <div className={`flex items-center gap-1 ${theme === 'forest' ? 'text-green-500' : theme === 'sunset' ? 'text-orange-500' : 'text-blue-500'}`}>
          <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <span>Data as of {new Date().toLocaleDateString()}</span>
        </div>
      </div>
      
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out forwards;
        }
      `}</style>
    </div>
  );
};

export default Charts;