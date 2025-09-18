import React, { useState, useMemo, useEffect, useRef } from "react";
import { 
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, AreaChart, Area, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Treemap, ComposedChart, ZAxis,
  Brush, ReferenceLine, Line
} from 'recharts';
import html2canvas from 'html2canvas';

// Backend endpoints
const SAMPLE_URL = "http://localhost:8000/api/sample-data/";
const STATUS_URL = "http://localhost:8000/api/prediction-status/";

// Helper: normalize backend payload
function normalizeApiPayload(payload) {
  if (!payload || typeof payload !== "object") return { results: [] };
  return {
    results: Array.isArray(payload.results) ? payload.results : [],
    alpha_diversity: payload.alpha_diversity || {},
    beta_diversity: payload.beta_diversity || {},
    rarefaction_curve: payload.rarefaction_curve || { x: [], y: [] },
    visualizations: payload.visualizations || payload.visualization_files || []
  };
}

const Charts = ({ data: externalData }) => {
  // Internal fetch state
  const [apiData, setApiData] = useState({ results: [] });
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  // Chart UI state
  const [chartType, setChartType] = useState('default');
  const [showCount, setShowCount] = useState(10);
  const [theme, setTheme] = useState('ocean');
  const [isAnimating, setIsAnimating] = useState(false);
  const [showInsights, setShowInsights] = useState(false);
  const chartRef = useRef(null);
  const pollRef = useRef(null);

  // Fetch initial data
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        setLoading(true);
        const res = await fetch(SAMPLE_URL);
        if (!res.ok) throw new Error("API " + res.status);
        const json = await res.json();
        if (cancelled) return;
        const norm = normalizeApiPayload(json);
        setApiData(norm);
        // If backend indicates still processing (no results yet) start polling status
        if (!norm.results.length && (json.status === 'pending' || json.processing)) {
          setProcessing(true);
        } else {
          setProcessing(false);
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, []);

  // Poll while processing
  useEffect(() => {
    if (!processing) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(STATUS_URL);
        const j = await r.json();
        if (j.status === 'ready') {
          setApiData(normalizeApiPayload(j));
          setProcessing(false);
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }
      } catch {
        /* silent */
      }
    }, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [processing]);

  // Allow external data override (e.g. parent passes fresh results after upload)
  useEffect(() => {
    if (externalData && Array.isArray(externalData)) {
      setApiData(d => ({ ...d, results: externalData }));
    }
  }, [externalData]);

  // Animate transitions
  useEffect(() => {
    setIsAnimating(true);
    const timer = setTimeout(() => setIsAnimating(false), 300);
    return () => clearTimeout(timer);
  }, [chartType, theme, showCount]);

  // Theme colors
  const COLORS = useMemo(() => {
    switch(theme) {
      case 'forest':
        return ['#047857','#10b981','#34d399','#6ee7b7','#a7f3d0','#064e3b','#065f46','#059669','#0d9488','#14b8a6'];
      case 'sunset':
        return ['#b91c1c','#dc2626','#ef4444','#f87171','#fca5a5','#7c2d12','#9a3412','#c2410c','#ea580c','#f97316'];
      default:
        return ['#2563eb','#3b82f6','#60a5fa','#93c5fd','#bfdbfe','#1d4ed8','#7c3aed','#8b5cf6','#a78bfa','#0ea5e9','#0284c7','#38bdf8'];
    }
  }, [theme]);

  // Records normalization (backend -> internal)
  const records = useMemo(() => {
    return (apiData.results || []).map(r => {
      const species = r.predicted_species || r.species || r.taxon || r.Taxonomy || "Unknown";
      const rawC = r.confidence ?? r.confidence_score ?? r.Confidence ?? r.score ?? 0;
      const conf = typeof rawC === 'number'
        ? (rawC > 1 ? rawC / 100 : rawC)
        : parseFloat(rawC) || 0;
      return {
        ...r,
        _species: species,
        _confidence: conf,          // normalized 0–1
        _status: (r.status || r.novelty || "").toLowerCase(),
        _cluster: r.cluster_id ?? r.cluster ?? r.clusterID ?? null
      };
    });
  }, [apiData.results]);

  // Species counts
  const speciesCounts = useMemo(() => {
    const acc = {};
    records.forEach(r => {
      acc[r._species] = (acc[r._species] || 0) + 1;
    });
    return acc;
  }, [records]);

  // Top species
  const chartData = useMemo(() => {
    return Object.entries(speciesCounts)
      .sort((a,b)=>b[1]-a[1])
      .slice(0, showCount)
      .map(([name,count], idx) => {
        const subset = records.filter(r => r._species === name);
        const avgConf = subset.reduce((s,r)=>s + r._confidence,0)/(subset.length||1);
        return {
          name: name.length>18? name.slice(0,18)+'…':name,
          fullName: name,
          count,
          rank: idx+1,
          confidence: avgConf
        };
      });
  }, [speciesCounts, showCount, records]);

  // Confidence distribution
  const confidenceData = useMemo(() => {
    const ranges = {'90-100%':0,'80-90%':0,'70-80%':0,'60-70%':0,'50-60%':0,'<50%':0};
    records.forEach(r => {
      const pct = r._confidence * 100;
      if (pct >= 90) ranges['90-100%']++;
      else if (pct >= 80) ranges['80-90%']++;
      else if (pct >= 70) ranges['70-80%']++;
      else if (pct >= 60) ranges['60-70%']++;
      else if (pct >= 50) ranges['50-60%']++;
      else ranges['<50%']++;
    });
    return Object.entries(ranges).map(([name,value])=>({
      name,
      value,
      percentage: records.length ? ((value/records.length)*100).toFixed(1) : "0.0"
    }));
  }, [records]);

  // Pie data
  const pieData = useMemo(() => {
    const topSum = chartData.reduce((s,d)=>s+d.count,0);
    const total = records.length;
    const other = total - topSum;
    return [...chartData, ...(other>0? [{ name:"Other", count: other, fullName:"Other Species"}] : [])];
  }, [chartData, records]);

  // Scatter data
  const scatterData = useMemo(() => {
    return Object.entries(speciesCounts).map(([name,count])=>{
      const subset = records.filter(r => r._species === name);
      const avgPct = subset.reduce((s,r)=>s + r._confidence*100,0)/(subset.length||1);
      return {
        name: name.length>18? name.slice(0,18)+'…':name,
        fullName: name,
        count,
        confidence: avgPct,
        size: Math.min(count*2,40)
      };
    });
  }, [speciesCounts, records]);

  // Novel vs known
  const novelVsKnown = useMemo(()=>{
    const novel = records.filter(r => r._status === 'novel').length;
    return [
      { name:'Novel', value: novel },
      { name:'Known', value: records.length - novel }
    ];
  }, [records]);

  // Trend (synthetic from confidence buckets)
  const trendData = useMemo(()=>{
    const bucketDefs = [
      { min:0.9, label:'90-100%' },
      { min:0.8, label:'80-90%' },
      { min:0.7, label:'70-80%' },
      { min:0.6, label:'60-70%' },
      { min:0.5, label:'50-60%' },
      { min:0.0, label:'<50%' }
    ];
    return bucketDefs.map((b,i)=>{
      const max = i===0 ? 1 : bucketDefs[i-1].min;
      const c = records.filter(r => r._confidence >= b.min && r._confidence < max).length;
      return {
        name: b.label,
        week1: Math.round(c*0.5),
        week2: Math.round(c*0.75),
        week3: c,
        avg: (c*0.5 + c*0.75 + c)/3
      };
    });
  }, [records]);

  // Insights
  const insights = useMemo(()=>{
    if (!records.length) return [];
    const out = [];
    if (chartData.length) {
      out.push({
        icon:'🔍',
        color:'text-blue-700',
        text:`Most common species "${chartData[0].fullName}" (${chartData[0].count}/${records.length}, ${(chartData[0].count/records.length*100).toFixed(1)}%).`
      });
    }
    const high = confidenceData.find(d=>d.name==='90-100%')?.value || 0;
    if (high) {
      out.push({
        icon:'✅',
        color:'text-green-700',
        text:`High-confidence detections: ${high} (${(high/records.length*100).toFixed(1)}%).`
      });
    }
    const novel = novelVsKnown[0].value;
    if (novel) {
      out.push({
        icon:'🧬',
        color:'text-purple-700',
        text:`Potential novel detections: ${novel} (${(novel/records.length*100).toFixed(1)}%).`
      });
    }
    out.push({
      icon:'🌿',
      color:'text-emerald-700',
      text:`Unique species: ${Object.keys(speciesCounts).length}.`
    });
    return out;
  }, [records, chartData, confidenceData, novelVsKnown, speciesCounts]);

  // Export chart
  const exportChart = async () => {
    if (!chartRef.current) return;
    try {
      const src = chartRef.current;

      // Build a clean clone (no Tailwind classes) with computed styles inlined
      const deepClone = src.cloneNode(true);

      // Helper: sanitize any unsupported color functions
      const sanitizeColorFns = (str) =>
        str.replace(/(oklab|oklch|lch|lab)\([^)]*\)/gi, "#3b82f6");

      // Inline computed styles to detach from original class rules
      const originalNodes = src.querySelectorAll("*");
      const cloneNodes = deepClone.querySelectorAll("*");

      const applyInline = (origEl, cloneEl) => {
        const cs = window.getComputedStyle(origEl);
        // Pick a subset of properties needed for visual fidelity
        const props = [
          "color","background","backgroundColor","backgroundImage",
          "border","borderColor","borderTopColor","borderRightColor",
          "borderBottomColor","borderLeftColor","boxShadow","fill","stroke",
          "font","fontSize","fontFamily","fontWeight","lineHeight",
          "padding","margin","display","alignItems","justifyContent",
          "flex","flexDirection","gap","width","height","minWidth","minHeight",
          "maxWidth","maxHeight","transform","opacity","overflow","textAlign"
        ];
        let inline = "";
        props.forEach(p => {
          let v = cs.getPropertyValue(p);
          if (!v) return;
          if (/oklab|oklch|lch|lab\(/i.test(v)) v = sanitizeColorFns(v);
          inline += `${p}:${v};`;
        });
        cloneEl.setAttribute("style", inline);
        // Remove class names to prevent pulling original stylesheet
        cloneEl.removeAttribute("class");
      };

      applyInline(src, deepClone);
      for (let i = 0; i < originalNodes.length; i++) {
        applyInline(originalNodes[i], cloneNodes[i]);
      }

      // Final HTML-level replace (handles any leftover inline style values)
      deepClone.innerHTML = sanitizeColorFns(deepClone.innerHTML);

      // Position clone off-screen
      deepClone.style.position = "fixed";
      deepClone.style.left = "-10000px";
      deepClone.style.top = "0";
      deepClone.style.boxShadow = "none";
      deepClone.classList.add("export-snapshot");

      document.body.appendChild(deepClone);

      const html2canvasLib = (await import("html2canvas")).default;
      const canvas = await html2canvasLib(deepClone, {
        backgroundColor: "#ffffff",
        scale: Math.min(2, window.devicePixelRatio || 2),
        logging: false,
        useCORS: true
      });

      document.body.removeChild(deepClone);

      const download = (blob) => {
        const a = document.createElement("a");
        const url = URL.createObjectURL(blob);
        a.href = url;
        a.download = `edna_chart_${chartType}_${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      };

      if (canvas.toBlob) {
        canvas.toBlob(b => b && download(b), "image/png", 0.92);
      } else {
        const dataUrl = canvas.toDataURL("image/png");
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = `edna_chart_${chartType}_${Date.now()}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (e) {
      console.error("Chart export failed:", e);
      alert("Export failed. Simplify chart styles and try again.");
    }
  };

  // Tooltips
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const row = payload[0].payload;
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-blue-100 text-xs">
          <p className="font-medium text-blue-800">{row.fullName || label}</p>
          {row.count !== undefined && <p className="text-blue-600">Count: {row.count}</p>}
          {row.confidence !== undefined && (
            <p className="text-green-600">
              Avg confidence: {(row.confidence<=1? row.confidence*100: row.confidence).toFixed(1)}%
            </p>
          )}
        </div>
      );
    }
    return null;
  };
  const CustomScatterTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const r = payload[0].payload;
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-blue-100 text-xs">
          <p className="font-medium text-blue-800">{r.fullName}</p>
          <p className="text-blue-600">Count: {r.count}</p>
          <p className="text-green-600">Confidence: {r.confidence.toFixed(1)}%</p>
        </div>
      );
    }
    return null;
  };

  // Chart renderer
  const renderChart = () => {
    switch(chartType) {
      case 'radar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart outerRadius={150} data={chartData.slice(0,8)}>
              <PolarGrid strokeDasharray="3 3" stroke="#bfdbfe" />
              <PolarAngleAxis dataKey="name" tick={{ fill:'#3b82f6', fontSize:12 }} />
              <PolarRadiusAxis stroke="#60a5fa" />
              <Radar dataKey="count" stroke={COLORS[0]} fill={COLORS[0]} fillOpacity={0.55}/>
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        );
      // case 'area':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="gCount" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={COLORS[0]} stopOpacity={0.8}/>
                  <stop offset="95%" stopColor={COLORS[0]} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#bfdbfe" />
              <XAxis dataKey="name" tick={{ fill:'#1e40af', fontSize:12 }} />
              <YAxis tick={{ fill:'#1e40af' }} />
              <Tooltip content={<CustomTooltip />} />
              <Area dataKey="count" stroke={COLORS[0]} fill="url(#gCount)" />
            </AreaChart>
          </ResponsiveContainer>
        );
      case 'confidence':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={confidenceData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#bfdbfe" horizontal={false}/>
              <XAxis type="number" tick={{ fill:'#1e40af' }} />
              <YAxis dataKey="name" type="category" tick={{ fill:'#1e40af', fontSize:12 }} width={90}/>
              <Tooltip />
              <Legend />
              <Bar dataKey="value" name="Sequences">
                {confidenceData.map((d,i)=><Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        );
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={70} outerRadius={120} paddingAngle={1}
                   dataKey="count" labelLine={false}
                   label={({name,percent}) => percent>0.05 ? `${name} ${(percent*100).toFixed(0)}%` : ""}>
                {pieData.map((_,i)=><Cell key={i} fill={COLORS[i % COLORS.length]} stroke="#fff"/>)}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );
      case 'treemap':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <Treemap data={chartData} dataKey="count" stroke="#fff">
              {chartData.map((_,i)=><Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Treemap>
          </ResponsiveContainer>
        );
      // case 'scatter':
      //   return (
      //     <ResponsiveContainer width="100%" height={400}>
      //       <ScatterChart>
      //         <CartesianGrid strokeDasharray="3 3" stroke="#bfdbfe" />
      //         <XAxis type="number" dataKey="count" name="Count" tick={{ fill:'#1e40af' }}/>
      //         <YAxis type="number" dataKey="confidence" name="Confidence %" tick={{ fill:'#1e40af' }}/>
      //         <ZAxis dataKey="size" range={[30,400]} />
      //         <Tooltip content={<CustomScatterTooltip />} />
      //         <Scatter data={scatterData} name="Species">
      //           {scatterData.map((_,i)=><Cell key={i} fill={COLORS[i % COLORS.length]} />)}
      //         </Scatter>
      //         <ReferenceLine y={80} stroke="green" strokeDasharray="3 3" />
      //         <ReferenceLine y={50} stroke="orange" strokeDasharray="3 3" />
      //       </ScatterChart>
      //     </ResponsiveContainer>
      //   );
      case 'novelty':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie data={novelVsKnown} innerRadius={80} outerRadius={140} dataKey="value" label>
                {novelVsKnown.map((_,i)=><Cell key={i} fill={COLORS[i % COLORS.length]} stroke="#fff"/>)}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        );
      case 'trend':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={trendData}>
              <CartesianGrid stroke="#bfdbfe" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fill:'#1e40af' }} />
              <YAxis tick={{ fill:'#1e40af' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="week1" fill={COLORS[0]} name="Week 1" />
              <Bar dataKey="week2" fill={COLORS[1]} name="Week 2" />
              <Bar dataKey="week3" fill={COLORS[2]} name="Week 3" />
              <Line dataKey="avg" stroke={COLORS[3]} name="Average" />
            </ComposedChart>
          </ResponsiveContainer>
        );
      default:
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top:20, right:30, left:10, bottom:70 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#bfdbfe" />
              <XAxis dataKey="name" tick={{ fill:'#1e40af', fontSize:12 }} interval={0} angle={-45} textAnchor="end"/>
              <YAxis tick={{ fill:'#1e40af' }} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Bar dataKey="count" name="Sequences">
                {chartData.map((_,i)=><Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        );
    }
  };

  if (error) {
    return (
      <div className="p-6 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm">
        Failed to load data: {error}
      </div>
    );
  }

  if (loading && !records.length) {
    return (
      <div className="p-6 rounded-xl bg-white/70 border border-blue-100 text-blue-700 text-sm">
        Loading data...
      </div>
    );
  }

  if (!records.length) {
    return (
      <div className="bg-gradient-to-br from-white to-blue-50/40 p-6 rounded-2xl shadow-lg border border-blue-100/50">
        <h2 className="text-xl font-bold bg-gradient-to-r from-blue-700 to-blue-500 bg-clip-text text-transparent mb-4">
          Data Visualization
        </h2>
        <div className="text-center py-10">
          <div className="text-blue-500 text-4xl mb-4">📊</div>
          <h3 className="text-lg font-medium text-blue-800 mb-2">
            No Visualization Data Available
          </h3>
          <p className="text-blue-600">
            {processing ? 'Processing predictions...' : 'Upload your eDNA samples to see visualizations'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gradient-to-br from-white to-blue-50/40 p-6 rounded-2xl shadow-lg border border-blue-100/50 transition-all duration-300`}>
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <h2 className="text-xl font-bold bg-gradient-to-r from-blue-700 to-blue-500 bg-clip-text text-transparent">
          eDNA Data Visualization
        </h2>
        <div className="flex flex-wrap gap-2">
          <select value={showCount} onChange={e=>setShowCount(Number(e.target.value))}
                  className="text-sm px-3 py-1 border rounded-lg bg-white/80 border-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option value={5}>Top 5</option><option value={10}>Top 10</option>
            <option value={15}>Top 15</option><option value={20}>Top 20</option>
          </select>
          <select value={theme} onChange={e=>setTheme(e.target.value)}
                  className="text-sm px-3 py-1 border rounded-lg bg-white/80 border-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500">
            <option value="ocean">Ocean</option>
            <option value="forest">Forest</option>
            <option value="sunset">Sunset</option>
          </select>
          <button onClick={exportChart}
                  className="text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white">
            Export
          </button>
        </div>
      </div>

      {/* Chart type selector (segmented control) */}
      <div className="mb-5">
        <div className="inline-flex flex-wrap items-center rounded-xl border border-blue-200 bg-gradient-to-br from-blue-50 to-blue-100 p-1 shadow-inner">
          {['default','pie','radar','confidence','treemap','novelty','trend'].map(t => {
            const active = chartType === t;
            return (
              <button
                key={t}
                onClick={() => setChartType(t)}
                aria-pressed={active}
                className={`text-xs font-medium px-3 py-2 rounded-lg transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
                  active
                    ? 'bg-white text-blue-700 shadow-sm'
                    : 'text-blue-700/80 hover:bg-blue-50'
                }`}
              >
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            );
          })}
        </div>
      </div>

      <div ref={chartRef}
           className={`bg-white/70 p-4 rounded-xl border border-blue-100 transition-opacity duration-300 ${isAnimating?'opacity-0':'opacity-100'}`}>
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-semibold text-blue-900">
            {
              {
                confidence:'Confidence Distribution',
                radar:'Species Radar Analysis',
                pie:'Species Distribution',
                treemap:'Species Proportion Treemap',
                novelty:'Novel vs Known Species',
                trend:'Temporal Trends (Synthetic)'
              }[chartType] || 'Top Species Breakdown'
            }
          </h3>
          <div className="text-xs px-2 py-0.5 rounded-full bg-blue-50 text-blue-600">
            Total: {records.length} sequences
          </div>
        </div>
        {renderChart()}
      </div>

      {/* Insights */}
      <div className="mt-4 border-t border-blue-100 pt-3">
        <button className="flex items-center gap-1 text-xs font-medium text-blue-600 mb-2"
                onClick={()=>setShowInsights(!showInsights)}>
          <span className={`transition-transform ${showInsights?'rotate-180':''}`}>▾</span>
          {showInsights ? 'Hide Insights' : 'Show Insights'}
        </button>
        {showInsights && insights.length>0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
            {insights.map((ins,i)=>(
              <div key={i} className="p-3 rounded-lg bg-white/60 border border-blue-100 flex gap-2 items-start">
                <div className="text-lg">{ins.icon}</div>
                <div className={`text-sm ${ins.color}`}>{ins.text}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {processing && (
        <div className="mt-4 text-xs text-blue-600 flex items-center gap-2">
          <span className="animate-pulse h-2 w-2 bg-blue-500 rounded-full" />
          Processing predictions... polling status
        </div>
      )}

      <div className="flex justify-between items-center text-xs text-gray-500 mt-4">
        <span></span>
        <span className="text-blue-500">Data as of {new Date().toLocaleDateString()}</span>
      </div>
    </div>
  );
};

export default Charts;