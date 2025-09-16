import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const Home = () => {
  const [scrollY, setScrollY] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Calculate color intensity based on scroll position
  const scrollProgress = Math.min(scrollY / 1000, 1);
  const lightBlue = `rgb(${165 - scrollProgress * 100}, ${180 - scrollProgress * 80}, ${255 - scrollProgress * 100})`;
  const darkBlue = `rgb(${30 + scrollProgress * 20}, ${58 + scrollProgress * 30}, ${138 + scrollProgress * 50})`;

  const handleDashboardClick = () => {
    navigate('/dashboard');
  };

  return (
    <div 
      className="relative min-h-[150vh] overflow-hidden"
      style={{
        background: `linear-gradient(to bottom, ${lightBlue} 0%, ${darkBlue} 100%)`,
        transition: 'background 0.3s ease'
      }}
    >
      {/* Floating Marine Animals */}
      <div className="fixed inset-0 pointer-events-none z-10">
        {/* Whale */}
        <div 
          className="absolute text-8xl animate-pulse marine-glow"
          style={{
            left: '10%',
            top: '20%',
            transform: `translateY(${scrollY * 0.2}px)`,
            opacity: 0.3 + scrollProgress * 0.4
          }}
        >
          🐋
        </div>
        
        {/* Jellyfish */}
        <div 
          className="absolute text-6xl marine-glow"
          style={{
            right: '15%',
            top: '35%',
            transform: `translateY(${scrollY * 0.15}px) rotate(${scrollY * 0.1}deg)`,
            opacity: 0.4 + scrollProgress * 0.3,
            animation: 'float 4s ease-in-out infinite'
          }}
        >
          🪼
        </div>
        
        {/* Octopus */}
        <div 
          className="absolute text-7xl marine-glow"
          style={{
            left: '70%',
            top: '60%',
            transform: `translateY(${scrollY * 0.25}px) translateX(${Math.sin(scrollY * 0.01) * 20}px)`,
            opacity: 0.3 + scrollProgress * 0.4
          }}
        >
          🐙
        </div>
        
        {/* Shark */}
        <div 
          className="absolute text-8xl marine-glow"
          style={{
            left: '20%',
            top: '80%',
            transform: `translateY(${scrollY * 0.3}px) scaleX(${1 + Math.sin(scrollY * 0.005) * 0.1})`,
            opacity: 0.2 + scrollProgress * 0.5
          }}
        >
          🦈
        </div>
        
        {/* Tropical Fish */}
        <div 
          className="absolute text-5xl marine-glow"
          style={{
            right: '25%',
            top: '15%',
            transform: `translateY(${scrollY * 0.18}px) translateX(${Math.cos(scrollY * 0.008) * 30}px)`,
            opacity: 0.4 + scrollProgress * 0.2
          }}
        >
          🐠
        </div>
        
        {/* Sea Turtle */}
        <div 
          className="absolute text-7xl marine-glow"
          style={{
            left: '5%',
            top: '70%',
            transform: `translateY(${scrollY * 0.12}px)`,
            opacity: 0.3 + scrollProgress * 0.4
          }}
        >
          🐢
        </div>
        
        {/* Dolphin */}
        <div 
          className="absolute text-6xl marine-glow"
          style={{
            right: '40%',
            top: '25%',
            transform: `translateY(${scrollY * 0.22}px) rotate(${Math.sin(scrollY * 0.003) * 10}deg)`,
            opacity: 0.3 + scrollProgress * 0.3
          }}
        >
          🐬
        </div>
        
        {/* Seahorse */}
        <div 
          className="absolute text-5xl marine-glow"
          style={{
            left: '85%',
            top: '45%',
            transform: `translateY(${scrollY * 0.16}px)`,
            opacity: 0.4 + scrollProgress * 0.2,
            animation: 'sway 3s ease-in-out infinite'
          }}
        >
          🦭
        </div>
      </div>

      {/* Floating animation bubbles */}
      <div className="fixed inset-0 pointer-events-none z-5">
        {[...Array(50)].map((_, i) => (
          <div
            key={i}
            className={`absolute rounded-full bubble-layer-${i % 4}`}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animation: `bubble ${3 + Math.random() * 6}s ease-in-out infinite ${Math.random() * 4}s`,
              transform: `translateY(${scrollY * (0.05 + Math.random() * 0.15)}px)`
            }}
          />
        ))}
      </div>

      {/* Main Content */}
      <div className="relative z-20 flex flex-col items-center justify-center min-h-screen px-6">
        {/* Hero Section */}
        <div className="max-w-3xl text-center">
          <h1 
            className="text-5xl font-bold mb-4 transition-colors duration-300 deep-text"
            style={{ color: scrollProgress > 0.3 ? '#e0f2fe' : '#1e40af' }}
          >
            eDNA Biodiversity Explorer
          </h1>
          <p 
            className="text-lg mb-8 transition-colors duration-300 deep-text"
            style={{ color: scrollProgress > 0.2 ? '#b3e5fc' : '#374151' }}
          >
            Analyze and visualize deep-sea biodiversity using environmental DNA (eDNA) datasets.
            Upload your processed CSV file to explore taxonomy insights, abundance patterns, 
            and community diversity in an interactive dashboard.
          </p>
          <button
            onClick={handleDashboardClick}
            className="pulse-glow bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-6 py-3 rounded-2xl shadow-lg transition-all hover:scale-105"
            style={{
              backgroundColor: scrollProgress > 0.4 ? '#0f172a' : '#4f46e5',
              boxShadow: scrollProgress > 0.4 ? '0 10px 25px rgba(15, 23, 42, 0.3)' : '0 10px 25px rgba(79, 70, 229, 0.3)'
            }}
          >
            Go to Dashboard
          </button>
        </div>

        {/* Feature Highlights */}
        <div className="grid md:grid-cols-3 gap-6 mt-16 max-w-5xl w-full">
          {[
            { title: "📂 Upload CSV", desc: "Upload eDNA result files in CSV format and process them instantly." },
            { title: "📊 Interactive Charts", desc: "Visualize biodiversity with rich interactive graphs and summary statistics." },
            { title: "⬇️ Export Data", desc: "Filter, explore, and download your results for further analysis or reporting." }
          ].map((card, i) => (
            <div 
              key={i}
              className="rounded-2xl p-6 transition-all duration-300 hover:scale-105 glass-card shimmer"
              style={{
                backgroundColor: scrollProgress > 0.3 ? 'rgba(30, 64, 175, 0.2)' : 'white',
                backdropFilter: scrollProgress > 0.3 ? 'blur(10px)' : 'none',
                border: scrollProgress > 0.3 ? '1px solid rgba(147, 197, 253, 0.3)' : 'none',
                boxShadow: scrollProgress > 0.3 ? '0 8px 32px rgba(30, 64, 175, 0.1)' : '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
            >
              <h3 
                className="text-xl font-semibold mb-2 transition-colors duration-300 deep-text"
                style={{ color: scrollProgress > 0.3 ? '#93c5fd' : '#1e40af' }}
              >
                {card.title}
              </h3>
              <p 
                className="transition-colors duration-300 deep-text"
                style={{ color: scrollProgress > 0.3 ? '#dbeafe' : '#4b5563' }}
              >
                {card.desc}
              </p>
            </div>
          ))}
        </div>

        {/* Additional scroll content */}
        <div className="mt-32 text-center max-w-2xl">
          <h2 
            className="text-3xl font-bold mb-6 transition-colors duration-300 deep-text"
            style={{ color: scrollProgress > 0.5 ? '#e0f2fe' : '#1e40af' }}
          >
            Dive Deeper into Marine Biodiversity
          </h2>
          <p 
            className="text-lg transition-colors duration-300 deep-text"
            style={{ color: scrollProgress > 0.4 ? '#b3e5fc' : '#374151' }}
          >
            As you scroll deeper into our ocean-themed interface, experience the transition 
            from shallow coastal waters to the mysterious depths of the deep sea. Our eDNA 
            analysis tools help researchers uncover the hidden biodiversity that exists in 
            these remote marine environments.
          </p>
        </div>
      </div>

      <style jsx>{`
        /* Marine Creature Glow */
        .marine-glow {
          filter: drop-shadow(0 0 10px rgba(173, 216, 230, 0.6));
          transition: transform 0.3s ease, filter 0.3s ease;
        }
        .marine-glow:hover {
          transform: scale(1.1);
          filter: drop-shadow(0 0 20px rgba(173, 216, 230, 0.9));
        }

        /* Deep glowing text */
        .deep-text {
          text-shadow: 0 0 8px rgba(147, 197, 253, 0.6);
        }

        /* Bubble Layers */
        .bubble-layer-0 { width: 8px; height: 8px; background: rgba(255,255,255,0.3); }
        .bubble-layer-1 { width: 5px; height: 5px; background: rgba(255,255,255,0.2); }
        .bubble-layer-2 { width: 3px; height: 3px; background: rgba(255,255,255,0.15); }
        .bubble-layer-3 { width: 2px; height: 2px; background: rgba(255,255,255,0.1); }

        /* Glassmorphism Cards */
        .glass-card {
          border-radius: 20px;
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
        }

        /* Shimmer effect */
        .shimmer {
          position: relative;
          overflow: hidden;
        }
        .shimmer::after {
          content: "";
          position: absolute;
          top: 0;
          left: -150%;
          width: 150%;
          height: 100%;
          background: linear-gradient(120deg, transparent, rgba(255,255,255,0.3), transparent);
          animation: shimmer 3s infinite;
        }

        /* Pulse Glow Button */
        .pulse-glow {
          animation: pulseGlow 2.5s infinite;
        }

        @keyframes pulseGlow {
          0%, 100% { box-shadow: 0 0 10px rgba(79,70,229,0.4), 0 0 20px rgba(79,70,229,0.2); }
          50% { box-shadow: 0 0 20px rgba(79,70,229,0.8), 0 0 40px rgba(79,70,229,0.4); }
        }

        @keyframes shimmer {
          0% { left: -150%; }
          50% { left: 100%; }
          100% { left: 100%; }
        }

        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(5deg); }
        }
        
        @keyframes sway {
          0%, 100% { transform: translateX(0px) rotate(0deg); }
          50% { transform: translateX(10px) rotate(2deg); }
        }
        
        @keyframes bubble {
          0% { transform: translateY(0px) scale(1); opacity: 0.1; }
          50% { opacity: 0.4; }
          100% { transform: translateY(-150px) scale(1.3); opacity: 0; }
        }
      `}</style>
    </div>
  );
};

export default Home;
