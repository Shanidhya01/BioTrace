import React, { useState, useEffect, useRef } from "react";
import UploadForm from "../components/UploadForm";
import SummaryCard from "../components/SummaryCards";
import TaxonomyTable from "../components/TaxonomyTable";
import DataTable from "../components/DataTable";
import Charts from "../components/Charts";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import ResultsSummary from '../components/ResultsSummary';

/* --- SVG Creatures (unchanged) --- */
const Fish = ({ className = "", flip = false, title = "fish" }) => (
  <svg
    className={className}
    width="120"
    height="60"
    viewBox="0 0 120 60"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    style={{ transform: flip ? "scaleX(-1)" : "none" }}
  >
    <title>{title}</title>
    <defs>
      <linearGradient id="g1" x1="0" x2="1">
        <stop offset="0" stopColor="#60a5fa" />
        <stop offset="1" stopColor="#1e3a8a" />
      </linearGradient>
      <linearGradient id="g2" x1="0" x2="1">
        <stop offset="0" stopColor="#a7f3d0" />
        <stop offset="1" stopColor="#34d399" />
      </linearGradient>
    </defs>
    <ellipse cx="60" cy="30" rx="36" ry="20" fill="url(#g1)" opacity="0.95" />
    <path d="M14 30 C26 18, 26 42, 14 30 Z" fill="url(#g2)" opacity="0.95" />
    <circle cx="80" cy="24" r="3.8" fill="#fff" opacity="0.95" />
    <path
      d="M90 28 Q100 30 106 26"
      stroke="#c7ddff"
      strokeWidth="1.6"
      strokeLinecap="round"
      fill="none"
      opacity="0.6"
    />
  </svg>
);

const Jellyfish = ({ className = "", title = "jellyfish" }) => (
  <svg
    className={className}
    width="80"
    height="110"
    viewBox="0 0 80 110"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <title>{title}</title>
    <defs>
      <linearGradient id="j1" x1="0" x2="1">
        <stop offset="0" stopColor="#93c5fd" stopOpacity="0.9" />
        <stop offset="1" stopColor="#60a5fa" stopOpacity="0.6" />
      </linearGradient>
    </defs>
    <path
      d="M40 12 C65 12, 72 36, 40 40 C8 36, 15 12, 40 12 Z"
      fill="url(#j1)"
    />
    <g opacity="0.9" stroke="#cfe9ff" strokeWidth="2" strokeLinecap="round">
      <path d="M30 44 C30 70, 20 84, 20 98" />
      <path d="M40 44 C40 70, 45 82, 45 98" />
      <path d="M50 44 C50 70, 60 84, 60 98" />
    </g>
  </svg>
);

const Dashboard = () => {
  // Always keep a stable shape
  const [data, setData] = useState({
    results: [],
    alpha_diversity: {},
    beta_diversity: {},
    rarefaction_curve: { x: [], y: [] },
    visualizations: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Convenience local variable with guards
  const results = Array.isArray(data.results) ? data.results : [];

  useEffect(() => {
    fetchData();
    const onScroll = () => {
      const depth = Math.min(window.scrollY / 800, 1);
      document.documentElement.style.setProperty("--ocean-depth", `${0.2 + depth * 0.6}`);
    };
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const normalizeApiPayload = (payload) => {
    if (!payload || typeof payload !== 'object') {
      return { results: [], alpha_diversity: {}, beta_diversity: {}, rarefaction_curve: { x: [], y: [] }, visualizations: [] };
    }
    return {
      results: Array.isArray(payload.results) ? payload.results : [],
      alpha_diversity: payload.alpha_diversity || {},
      beta_diversity: payload.beta_diversity || {},
      rarefaction_curve: payload.rarefaction_curve || { x: [], y: [] },
      visualizations: payload.visualizations || payload.visualization_files || []
    };
  };

  const fetchData = async () => {
    try {
      setLoading(true);
      const response = await fetch("http://localhost:8000/api/sample-data/");
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      const jsonData = await response.json();
      setData(normalizeApiPayload(jsonData));
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(err.message || "Fetch failed");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (uploadedData) => {
    // If your UploadForm already returns API-shaped data:
    setData(normalizeApiPayload(uploadedData));
  };

  const handleDownload = (rows) => {
    const list = Array.isArray(rows) ? rows : [];
    if (!list.length) return;
    const headers = ["Predicted Species","Confidence","Status","Cluster ID","Top Predictions"];
    const csvRows = list.map(item => {
      const conf = (item.confidence ?? item.confidence_score ?? 0) * 100;
      const tops = Array.isArray(item.top_predictions)
        ? item.top_predictions.slice(0,3).map(tp =>
            `${tp.species} (${((tp.confidence ?? tp.confidence_score ?? 0)*100).toFixed(2)}%)`
          ).join("; ")
        : "";
      return [
        (item.predicted_species || "").replace(/,/g," "),
        conf.toFixed(2)+"%",
        (item.status || "").replace(/,/g," "),
        (item.cluster_id ?? "").toString().replace(/,/g," "),
        tops.replace(/,/g,";")
      ];
    });
    const csv = [headers.join(","), ...csvRows.map(r=>r.join(","))].join("\n");
    const blob = new Blob([csv], { type:"text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "prediction_results.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const getSummaryStats = () => {
    const speciesSet = new Set(results.map(r => r.predicted_species));
    const sampleSet = new Set(results.map(r => r.sample_id));
    const novelClusters = new Set(
      results.filter(r => (r.status || "").toLowerCase() === "novel")
             .map(r => r.cluster_id)
    );
    return {
      totalSequences: results.length,
      uniqueSpecies: speciesSet.size,
      totalSamples: sampleSet.size,
      novelClusters: novelClusters.size,
      alpha_diversity: data.alpha_diversity || {},
      beta_diversity: data.beta_diversity || {}
    };
  };

  const summaryStats = getSummaryStats();
  const dashboardRef = useRef();

  const downloadPDF = async () => {
    try {
      const input = dashboardRef.current;
      if (!input) return;
      const canvas = await html2canvas(input, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#ffffff"
      });
      const img = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p","mm","a4");
      const pw = pdf.internal.pageSize.getWidth();
      const ph = pdf.internal.pageSize.getHeight();
      const imgH = (canvas.height * pw) / canvas.width;
      let heightLeft = imgH;
      let position = 0;
      pdf.addImage(img, "PNG", 0, position, pw, imgH);
      heightLeft -= ph;
      while (heightLeft > 0) {
        position = heightLeft - imgH;
        pdf.addPage();
        pdf.addImage(img, "PNG", 0, position, pw, imgH);
        heightLeft -= ph;
      }
      pdf.save("dashboard.pdf");
    } catch (e) {
      console.error("PDF export failed:", e);
    }
  };

  /* Creature configs (unchanged) */
  const creatures = [
    {
      id: "fish-1",
      type: "fish",
      top: "20%",
      delay: 0,
      duration: 18,
      size: 120,
      flip: false,
      z: 5,
    },
    {
      id: "fish-2",
      type: "fish",
      top: "65%",
      delay: 4,
      duration: 26,
      size: 90,
      flip: true,
      z: 6,
    },
    {
      id: "fish-3",
      type: "fish",
      top: "40%",
      delay: 8,
      duration: 22,
      size: 100,
      flip: false,
      z: 4,
    },
    {
      id: "jelly-1",
      type: "jelly",
      top: "50%",
      delay: 2,
      duration: 28,
      size: 80,
      z: 3,
    },
    {
      id: "jelly-2",
      type: "jelly",
      top: "72%",
      delay: 10,
      duration: 34,
      size: 100,
      z: 4,
    },
  ];

  return (
    <div
      className="relative min-h-screen overflow-hidden"
      style={{
        background:
          "radial-gradient(circle at 10% 10%, rgba(99,102,241,0.06), transparent 100px), linear-gradient(180deg,#c7f3ff 0%, #60a5fa 20%, #1e3a8a 85%)",
      }}
      ref={dashboardRef}
    >
      {/* (Optional) PDF download button
      <div className="flex justify-end px-6 py-4">
        <button onClick={downloadPDF} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow-lg transition">
          Download Dashboard as PDF
        </button>
      </div>
      */}

      {/* Background / creatures (unchanged) */}
      <div className="absolute inset-0 bubble-large pointer-events-none z-0" />
      <div className="absolute inset-0 bubble-medium pointer-events-none z-0" />
      <div className="absolute inset-0 bubble-small pointer-events-none z-0" />
      <div className="absolute inset-0 bubble-micro pointer-events-none z-0" />

      {/* Light rays */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="light-rays" />
      </div>

      {/* Animated creatures (interactive) */}
      <div className="absolute inset-0 z-20 pointer-events-none">
        {creatures.map((c) => {
          const style = {
            top: c.top,
            left: "-18%",
            width: `${c.size}px`,
            transform: c.flip ? "scaleX(-1)" : "none",
            zIndex: c.z + 100,
            animationDelay: `${c.delay}s`,
            animationDuration: `${c.duration}s`,
          };
          const wrapperClass =
            "creature-wrapper pointer-events-auto absolute -translate-x-0 will-change-transform";
          return (
            <div
              key={c.id}
              className={wrapperClass}
              style={style}
              // make hovering possible on creatures
              onMouseEnter={(e) =>
                e.currentTarget.classList.add("creature-hover")
              }
              onMouseLeave={(e) =>
                e.currentTarget.classList.remove("creature-hover")
              }
            >
              {c.type === "fish" ? (
                <Fish className="creature-svg" flip={c.flip} title={c.id} />
              ) : (
                <Jellyfish className="creature-svg" title={c.id} />
              )}
            </div>
          );
        })}
      </div>

      {/* Header (unchanged) */}
      <div className="relative overflow-hidden bg-white/90 backdrop-blur-md border-b border-blue-200/50 shadow-2xl z-30">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 via-blue-500/10 to-blue-700/20" />
        <div className="relative px-6 py-10">
          <div className="max-w-7xl mx-auto flex items-center space-x-4">
            <div className="p-3 bg-gradient-to-br from-blue-400 to-blue-600 rounded-2xl shadow-lg transform hover:scale-110 transition z-40">
              <span className="text-3xl animate-pulse">🌊</span>
            </div>
            <div>
              <h1 className="text-4xl font-extrabold bg-gradient-to-r from-blue-600 via-blue-700 to-blue-900 bg-clip-text text-transparent drop-shadow-lg">
                Biodiversity Dashboard
              </h1>
              <p className="text-blue-800 mt-2 font-medium tracking-wide shimmer-text z-40">
                Analyze and visualize ecological data with precision
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-30 max-w-7xl mx-auto px-6 py-8 space-y-8">
        <div className="transform transition-all duration-500 hover:scale-[1.02]">
          <div className="bg-white/85 backdrop-blur-lg rounded-2xl shadow-2xl border border-blue-300/40 p-6 hover:shadow-blue-300/40 transition pulse-glow">
            <UploadForm onUpload={handleUpload} />
          </div>
        </div>

        {error && (
          <div className="bg-red-100/90 backdrop-blur-sm border border-red-300 rounded-2xl p-4 text-red-800 shadow-xl">
            {error}
          </div>
        )}

        {loading && (
          <div className="flex flex-col items-center justify-center py-20 space-y-6">
            <div className="relative">
              <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-200" />
              <div className="absolute inset-0 animate-ping rounded-full h-20 w-20 border-4 border-t-blue-500 border-r-blue-700" />
            </div>
            <div className="text-center space-y-2">
              <p className="text-lg font-semibold text-white drop-shadow">
                Processing data...
              </p>
              <p className="text-sm text-blue-200">
                Analyzing biodiversity patterns
              </p>
            </div>
          </div>
        )}

        {!loading && !error && results.length > 0 && (
          <div className="space-y-10 animate-fadeIn">
            <section>
              <h2 className="section-title">Overview</h2>
              <div className="glass-card">
                <ResultsSummary data={{
                  ...data,
                  results, // ensure normalized
                  summary: summaryStats
                }} />
              </div>
            </section>

            <section>
              <h2 className="text-xl font-semibold text-blue-900 mb-3">
                Taxonomic Analysis
              </h2>
              <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-4 shadow-lg hover:bg-white/10 transition-all duration-300">
                <TaxonomyTable data={results} />
              </div>
            </section>

            <section>
              <h2 className="section-title">Data Visualization</h2>
              <div className="glass-card">
                <Charts
                  data={results}
                  alphaDiversity={data.alpha_diversity}
                  betaDiversity={data.beta_diversity}
                  rarefactionCurve={data.rarefaction_curve}
                />
              </div>
            </section>

            <section>
              <h2 className="section-title">Prediction Results</h2>
              <div className="flex justify-end mb-3">
                <button
                  onClick={() => handleDownload(results)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow hover:bg-blue-700 transition"
                >
                  ⬇ Download Results
                </button>
              </div>
              <div className="glass-table">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-blue-200">
                    <thead className="bg-blue-50/60 backdrop-blur-sm">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-blue-700 uppercase tracking-wider">Predicted Species</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-blue-700 uppercase tracking-wider">Confidence</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-blue-700 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-blue-700 uppercase tracking-wider">Cluster ID</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-blue-700 uppercase tracking-wider">Top 3 Predictions</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white/50 backdrop-blur-md divide-y divide-blue-200">
                      {results.map((item, i) => {
                        const conf = (item.confidence ?? item.confidence_score ?? 0) * 100;
                        const tops = Array.isArray(item.top_predictions) ? item.top_predictions.slice(0,3) : [];
                        return (
                          <tr key={i} className="hover:bg-blue-50/50 transition-colors duration-200">
                            <td className="px-6 py-4 text-sm text-blue-900 font-medium">
                              {item.predicted_species || '—'}
                            </td>
                            <td className="px-6 py-4 text-sm text-blue-800">
                              <span className="bg-blue-100/70 text-blue-800 px-2 py-1 rounded-lg font-semibold shadow-sm">
                                {conf.toFixed(2)}%
                              </span>
                            </td>
                            <td className="px-6 py-4 text-sm">
                              <span className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full shadow-sm ${
                                (item.status || '').toLowerCase() === "novel"
                                  ? "bg-amber-100/80 text-amber-800 border border-amber-200/60"
                                  : "bg-emerald-100/80 text-emerald-800 border border-emerald-200/60"
                              }`}>
                                {item.status || 'unknown'}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-sm text-blue-800 font-mono">
                              {item.cluster_id ?? '—'}
                            </td>
                            <td className="px-6 py-4 text-sm text-blue-700">
                              {tops.length
                                ? tops.map((tp, idx2) => (
                                    <div key={idx2}>
                                      {tp.species} ({((tp.confidence ?? tp.confidence_score ?? 0)*100).toFixed(2)}%)
                                    </div>
                                  ))
                                : <span className="text-blue-400 italic">n/a</span>}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          </div>
        )}

        {!loading && !error && results.length === 0 && (
          <div className="text-center py-16 space-y-6 relative z-10">
            <div className="mx-auto w-24 h-24 bg-gradient-to-br from-blue-200/80 to-blue-400/80 backdrop-blur-md rounded-full flex items-center justify-center shadow-2xl border border-blue-300/50 animate-bounce">
              <span className="text-5xl">🌊</span>
            </div>
            <div className="space-y-2">
              <h3 className="text-2xl font-semibold text-white drop-shadow-xl">
                Ready to explore biodiversity data
              </h3>
              <p className="text-blue-100 max-w-md mx-auto drop-shadow">
                Upload your CSV file to begin analyzing species distribution and ecological insights.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Styles (existing) */}
      <style jsx>{`
        :root {
          --ocean-depth: 0.4;
        }

        /* Fade in */
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.8s ease-out forwards;
        }

        /* Shimmer text */
        .shimmer-text {
          background: linear-gradient(90deg, #1e3a8a, #60a5fa, #1e3a8a);
          background-size: 200% auto;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          animation: shimmer 3s linear infinite;
        }
        @keyframes shimmer {
          0% {
            background-position: 200% center;
          }
          100% {
            background-position: 0 center;
          }
        }

        /* Pulse glow for cards */
        .pulse-glow {
          animation: glowPulse 4s infinite;
        }
        @keyframes glowPulse {
          0%,
          100% {
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.45),
              0 0 30px rgba(59, 130, 246, 0.25);
          }
          50% {
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.75),
              0 0 40px rgba(59, 130, 246, 0.45);
          }
        }

        /* Glass cards */
        .glass-card {
          background: rgba(255, 255, 255, calc(0.8 * var(--ocean-depth) + 0.2));
          backdrop-filter: blur(12px);
          border: 1px solid rgba(59, 130, 246, 0.25);
          border-radius: 1rem;
          padding: 1.5rem;
          transition: all 0.3s ease;
        }
        .glass-card:hover {
          box-shadow: 0 10px 30px rgba(2, 6, 23, 0.25);
        }

        .section-title {
          font-size: 1.5rem;
          font-weight: 700;
          color: #fff;
          text-shadow: 0 5px 20px rgba(0, 0, 0, 0.55);
          margin-bottom: 1rem;
        }

        /* Bubble layers - background tile approach */
        .bubble-large::before,
        .bubble-medium::before,
        .bubble-small::before,
        .bubble-micro::before {
          content: "";
          position: absolute;
          inset: -20% -20% -40% -20%;
          background-repeat: repeat;
          opacity: calc(0.25 * var(--ocean-depth) + 0.2);
        }
        .bubble-large::before {
          background-image: radial-gradient(
            circle,
            rgba(255, 255, 255, 0.42) 8px,
            transparent 9px
          );
          background-size: 220px 220px;
          animation: riseLarge 80s linear infinite;
        }
        .bubble-medium::before {
          background-image: radial-gradient(
            circle,
            rgba(255, 255, 255, 0.28) 6px,
            transparent 6.5px
          );
          background-size: 140px 140px;
          animation: riseMedium 60s linear infinite;
        }
        .bubble-small::before {
          background-image: radial-gradient(
            circle,
            rgba(255, 255, 255, 0.18) 4px,
            transparent 4.5px
          );
          background-size: 90px 90px;
          animation: riseSmall 42s linear infinite;
        }
        .bubble-micro::before {
          background-image: radial-gradient(
            circle,
            rgba(255, 255, 255, 0.12) 2px,
            transparent 2.5px
          );
          background-size: 48px 48px;
          animation: riseMicro 30s linear infinite;
        }

        @keyframes riseLarge {
          0% {
            transform: translateY(0) translateX(0) rotate(0deg);
          }
          100% {
            transform: translateY(-250px) translateX(120px) rotate(30deg);
          }
        }
        @keyframes riseMedium {
          0% {
            transform: translateY(0) translateX(0) rotate(0deg);
          }
          100% {
            transform: translateY(-200px) translateX(-100px) rotate(-20deg);
          }
        }
        @keyframes riseSmall {
          0% {
            transform: translateY(0) translateX(0) rotate(0deg);
          }
          100% {
            transform: translateY(-150px) translateX(80px) rotate(18deg);
          }
        }
        @keyframes riseMicro {
          0% {
            transform: translateY(0) translateX(0) rotate(0deg);
          }
          100% {
            transform: translateY(-100px) translateX(-60px) rotate(-12deg);
          }
        }

        /* Light rays */
        .light-rays {
          width: 120%;
          height: 140%;
          margin-left: -10%;
          background: repeating-linear-gradient(
            -35deg,
            rgba(255, 255, 255, 0.04) 0px,
            rgba(255, 255, 255, 0.04) 3px,
            transparent 3px,
            transparent 12px
          );
          transform: skewX(-10deg);
          mix-blend-mode: overlay;
          animation: raysMove 16s linear infinite;
          opacity: calc(0.6 * var(--ocean-depth));
        }
        @keyframes raysMove {
          0% {
            transform: translateX(0) skewX(-10deg);
          }
          100% {
            transform: translateX(220px) skewX(-10deg);
          }
        }

        /* Creature base behavior */
        .creature-wrapper {
          left: -22%;
          will-change: transform, opacity;
          transition: transform 300ms ease, filter 300ms ease;
          animation-name: swim;
          animation-timing-function: linear;
          animation-iteration-count: infinite;
        }

        /* swim moves creature horizontally across screen while also providing vertical sine-like wobble using keyframes */
        @keyframes swim {
          0% {
            transform: translateX(-10vw) translateY(0);
            opacity: 0;
          }
          6% {
            opacity: 1;
          }
          50% {
            transform: translateX(110vw) translateY(-26px);
          }
          100% {
            transform: translateX(240vw) translateY(0);
            opacity: 0;
          }
        }

        /* Small vertical oscillation to emulate sine */
        .creature-svg {
          display: block;
          transition: transform 220ms ease, filter 220ms ease;
          filter: drop-shadow(0 6px 20px rgba(30, 58, 138, 0.18));
        }

        /* Hover interactions: scale + glow + pause (by pausing animation) */
        .creature-wrapper.creature-hover {
          transform: scale(1.07);
          filter: drop-shadow(0 18px 40px rgba(96, 165, 250, 0.22));
          z-index: 9999 !important;
          animation-play-state: paused;
        }
        .creature-wrapper.creature-hover .creature-svg {
          transform-origin: center;
          transform: scale(1.06) translateY(-6px);
        }

        /* Jelly wobble */
        @keyframes jellyPulse {
          0% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-10px) scaleY(0.98);
          }
          100% {
            transform: translateY(0);
          }
        }
        /* Add subtle pulsing to jellyfish shapes */
        .creature-wrapper .creature-svg path,
        .creature-wrapper .creature-svg ellipse {
          animation: jellyPulse 3.6s ease-in-out infinite;
        }

        /* Make creatures responsive and less intrusive on small screens */
        @media (max-width: 640px) {
          .creature-wrapper {
            display: none;
          } /* hide on tiny screens to conserve space */
        }
      `}</style>
    </div>
  );
};

export default Dashboard;
