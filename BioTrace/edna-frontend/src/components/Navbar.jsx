import React from "react";
import { Link, useLocation } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();
  const currentPath = location.pathname;

  // Add a small school of fishes with varied sizes/speeds/directions
  const fishes = [
    { top: "20%", size: 18, color: "#3b82f6", duration: 11, delay: 0.0, reverse: false },
    { top: "32%", size: 14, color: "#60a5fa", duration: 9,  delay: 0.6, reverse: true  },
    { top: "45%", size: 20, color: "#2563eb", duration: 13, delay: 1.2, reverse: false },
    { top: "58%", size: 16, color: "#38bdf8", duration: 10, delay: 1.8, reverse: true  },
    { top: "70%", size: 15, color: "#0ea5e9", duration: 12, delay: 0.9, reverse: false },
    { top: "26%", size: 13, color: "#22d3ee", duration: 8,  delay: 1.5, reverse: true  },
  ];

  const Button = ({ variant, className, children, to }) => {
    const baseClasses =
      "px-4 py-2 font-medium transition-all duration-300 shadow-md hover:shadow-xl transform hover:-translate-y-0.5 rounded-xl relative overflow-hidden";
    const variantClasses =
      variant === "default"
        ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white border-0 hover:from-blue-700 hover:to-cyan-700"
        : "border-2 border-blue-300 text-blue-700 hover:bg-blue-50 hover:border-blue-400 bg-white/70 backdrop-blur-md";
    return (
      <Link to={to} className={`${baseClasses} ${variantClasses} ${className}`}>
        <span className="relative z-10">{children}</span>
        {/* Button shimmer */}
        <span className="absolute inset-0 bg-gradient-to-r from-white/10 via-white/30 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-500"></span>
      </Link>
    );
  };

  return (
    <nav className="relative bg-gradient-to-r from-blue-50/70 via-cyan-50/70 to-blue-100/70 shadow-xl px-6 py-4 flex justify-between items-center sticky top-0 z-50 backdrop-blur-lg border-b border-blue-200/40 overflow-hidden">
      {/* removed bottom wave bar */}

      {/* Bubbles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-blue-200 opacity-40"
            style={{
              left: `${10 + Math.random() * 80}%`,
              width: `${2 + Math.random() * 4}px`,
              height: `${2 + Math.random() * 4}px`,
              bottom: "-10px",
              animation: "bubble 6s ease-in-out infinite",
              animationDelay: `${i * 0.5}s`,
            }}
          />
        ))}
      </div>

      {/* Marine SVG creatures */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Multiple fishes */}
        {fishes.map((f, i) => (
          <svg
            key={i}
            className="absolute"
            style={{
              top: f.top,
              left: f.reverse ? "110%" : "-10%",
              width: `${f.size}px`,
              height: `${f.size}px`,
              color: f.color,
              animation: `${f.reverse ? "swim-reverse" : "swim"} ${f.duration}s linear infinite`,
              animationDelay: `${f.delay}s`,
              opacity: 0.9,
              filter: "drop-shadow(0 1px 1px rgba(0,0,0,0.08))",
            }}
            viewBox="0 0 24 24"
            fill="currentColor"
          >
            <path d="M2 12s4-5 10-5c6 0 10 5 10 5s-4 5-10 5c-6 0-10-5-10-5z" />
            <circle cx="8" cy="12" r="1" fill="white" />
          </svg>
        ))}

        {/* Turtle (kept) */}
        <svg
          className="absolute"
          style={{
            top: "62%",
            left: "110%",
            width: "22px",
            height: "22px",
            color: "#22c55e",
            animation: "swim-reverse 14s linear infinite",
          }}
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <circle cx="12" cy="12" r="5" />
          <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2" />
        </svg>

        {/* jellyfish removed */}
      </div>

      {/* Logo */}
      <Link
        to="/"
        className="text-2xl font-bold bg-gradient-to-r from-blue-600 via-cyan-600 to-blue-700 bg-clip-text text-transparent relative z-10 drop-shadow-sm hover:scale-105 transition-transform duration-300"
      >
        eDNA Explorer
      </Link>

      {/* Buttons */}
      <div className="flex gap-4 relative z-10">
        <Button variant={currentPath === "/" ? "default" : "outline"} to="/">
          Home
        </Button>
        <Button
          variant={currentPath === "/dashboard" ? "default" : "outline"}
          to="/dashboard"
        >
          Dashboard
        </Button>
      </div>

      <style>{`
        @keyframes bubble {
          0% { transform: translateY(0) scale(1); opacity: .3; }
          50% { opacity: .6; }
          100% { transform: translateY(-60px) scale(1.2); opacity: 0; }
        }
        @keyframes swim {
          0% { transform: translateX(0); }
          100% { transform: translateX(120%); }
        }
        @keyframes swim-reverse {
          0% { transform: translateX(0); }
          100% { transform: translateX(-120%); }
        }
        @keyframes floaty {
          0%,100% { transform: translateY(0); }
          50% { transform: translateY(-6px); }
        }
      `}</style>
    </nav>
  );
};

export default Navbar;
