import React from "react";
import { Link, useLocation } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();
  const currentPath = location.pathname;

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
      {/* Ocean wave */}
      <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-500 opacity-70"></div>

      {/* Bubbles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-blue-200 opacity-40 animate-bubble"
            style={{
              left: `${10 + Math.random() * 80}%`,
              width: `${2 + Math.random() * 4}px`,
              height: `${2 + Math.random() * 4}px`,
              bottom: "-10px",
              animationDuration: `${4 + Math.random() * 6}s`,
              animationDelay: `${i * 0.5}s`,
            }}
          ></div>
        ))}
      </div>

      {/* Marine SVG creatures (inside navbar only) */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Fish */}
        <svg
          className="absolute w-6 h-6 text-blue-500"
          style={{
            top: "40%",
            left: "-10%",
            animation: "swim 10s linear infinite",
          }}
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M2 12s4-5 10-5c6 0 10 5 10 5s-4 5-10 5c-6 0-10-5-10-5z" />
          <circle cx="8" cy="12" r="1" fill="white" />
        </svg>

        {/* Turtle */}
        <svg
          className="absolute w-7 h-7 text-green-500"
          style={{
            top: "60%",
            left: "110%",
            animation: "swim-reverse 14s linear infinite",
          }}
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <circle cx="12" cy="12" r="5" />
          <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2" />
        </svg>

        {/* Jellyfish */}
        <svg
          className="absolute w-5 h-5 text-pink-400"
          style={{
            top: "20%",
            left: "50%",
            animation: "floaty 6s ease-in-out infinite",
          }}
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <circle cx="12" cy="8" r="4" />
          <path d="M10 12v4M14 12v4M8 13v3M16 13v3" stroke="currentColor" />
        </svg>
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
          0% {
            transform: translateY(0) scale(1);
            opacity: 0.3;
          }
          50% {
            opacity: 0.6;
          }
          100% {
            transform: translateY(-60px) scale(1.2);
            opacity: 0;
          }
        }
        @keyframes swim {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(120%);
          }
        }
        @keyframes swim-reverse {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-120%);
          }
        }
        @keyframes floaty {
          0%,
          100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-6px);
          }
        }
      `}</style>
    </nav>
  );
};

export default Navbar;
