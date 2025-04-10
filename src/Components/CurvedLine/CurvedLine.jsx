import React from "react";

const CurvedLine = () => {
  return (
    <div className="relative w-full h-[100px] bg-black overflow-hidden">
      <svg className="absolute top-0 left-0 w-full h-full" viewBox="0 0 1440 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="curveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#1e3a8a" /> {/* Dark Blue */}
            <stop offset="100%" stopColor="#dc2626" /> {/* Red */}
          </linearGradient>
        </defs>
        <path d="M0,80 C480,120 960,40 1440,80 L1440,100 L0,100 Z" fill="url(#curveGradient)" />
      </svg>
    </div>
  );
};

export default CurvedLine;