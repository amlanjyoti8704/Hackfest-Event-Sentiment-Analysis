import React from 'react'
import { useState } from "react";
import { FaLinkedin, FaInstagram } from "react-icons/fa";

const developers = [
    { 
      name: "Sahasranshu Behera", 
      role: "Tech Team Lead", 
      img: "../src/assets/image copy 2.png", 
      linkedin: "https://www.linkedin.com/in/sahasranshu-behera-b961a519a/", 
      instagram: "https://www.instagram.com/somuwuu/", 
      description: `I’m Sahasranshu Behera, a passionate software developer with a deep interest in AI and Machine Learning. I constantly learn and innovate, exploring diverse projects to harness technology for meaningful impact.`
    },
    { 
        name: "Amlan Jyoti", 
        role: "Frontend Developer", 
        img: "../src/assets/image copy.png", 
        linkedin: "https://www.linkedin.com/in/amlan-jyoti-a0013717b/", 
        instagram: "https://www.instagram.com/amlan._.jyoti/", 
        description: `I’m Amlan Jyoti, a passionate software developer with a deep interest in AI and Machine Learning. I constantly learn and innovate, exploring diverse projects to harness technology for meaningful impact.`
      },
      // { 
      //   name: "Abhas Singh", 
      //   role: "Backend Developer", 
      //   img: "../src/assets/image copy.png", 
      //   linkedin: "https://www.linkedin.com/in/manoj-kumar-1a3765254/", 
      //   instagram: "https://www.instagram.com/infinite_realm_?igsh=dW5jZHdsZzV0ZG12", 
      //   description: `I’m Abhas Singh, a passionate software developer with a deep interest in AI and Machine Learning. I constantly learn and innovate, exploring diverse projects to harness technology for meaningful impact.`
      // },
      // { 
      //   name: "Yash Shekhar", 
      //   role: "UI-UX Designer", 
      //   img: "../src/assets/image copy.png", 
      //   linkedin: "https://www.linkedin.com/in/manoj-kumar-1a3765254/", 
      //   instagram: "https://www.instagram.com/infinite_realm_?igsh=dW5jZHdsZzV0ZG12", 
      //   description: `I’m Yash Shekhar, a passionate software developer with a deep interest in AI and Machine Learning. I constantly learn and innovate, exploring diverse projects to harness technology for meaningful impact.`
      // },
      // { 
      //   name: "Shubham Raj Verma", 
      //   role: "Backend Developer", 
      //   img: "../src/assets/image copy.png", 
      //   linkedin: "https://www.linkedin.com/in/manoj-kumar-1a3765254/", 
      //   instagram: "https://www.instagram.com/infinite_realm_?igsh=dW5jZHdsZzV0ZG12", 
      //   description: `I’m Shubham Raj Verma, a passionate software developer with a deep interest in AI and Machine Learning. I constantly learn and innovate, exploring diverse projects to harness technology for meaningful impact.`
      // },
  ]

export default function DevelopersPage() {
    const [flippedIndex, setFlippedIndex] = useState(null);
  
    const handleFlip = (index) => {
      setFlippedIndex(flippedIndex === index ? null : index);
    };
  
    return (
      <div className="relative min-h-screen flex flex-col items-center justify-center py-20 px-5 bg-gradient-to-br from-[#0A0118] via-[#2D1E0F] to-[#1A0B2E] overflow-hidden">
        {/* Background effects */}
        <div className="absolute inset-0 bg-black opacity-100 pointer-events-none"></div>
        <div className="absolute top-0 -left-4 w-96 h-96 bg-black rounded-full mix-blend-normal filter blur-[128px] opacity-30 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-96 h-96 bg-gray-900 rounded-full mix-blend-normal filter blur-[128px] opacity-30 animate-blob animation-delay-2000"></div>
        
        {/* Heading */}
        <h1 className="text-6xl md:text-7xl text-center font-bold tracking-tight mb-10 sm:mb-20 animate-fade-in z-10">
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-slate-700 via-slate-500 to-slate-800 animate-gradient-x">
            Developers Team
          </span>
        </h1>
  
        {/* Developer Cards */}
        <div className="relative z-10 max-w-7xl w-full mx-auto px-4">
          {/* <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8 justify-items-center"> */}
          <div className="flex flex-wrap justify-center gap-8 max-w-7xl w-full mx-auto px-4">
            {developers.map((dev, index) => (
              <div
                key={index}
                className="group h-[470px] w-[300px] [perspective:1000px] cursor-pointer"
                onClick={() => handleFlip(index)}
              >
                <div
                  className={`relative w-full h-full transition-all duration-700 [transform-style:preserve-3d] ${flippedIndex === index ? '[transform:rotateY(180deg)]' : ''}`}
                >
                  {/* Front Side */}
                  <div className="absolute inset-0 rounded-2xl">
                    <div className="relative h-full bg-gradient-to-br from-black/40 to-black/60 backdrop-blur-xl border border-slate-300/20 rounded-2xl p-6 overflow-hidden group-hover:border-gray-300/40 transition-all duration-300">
                      {/* Gradient Overlay */}
                      <div className="absolute inset-0 bg-gradient-to-b from-blue-300/10 via-transparent to-gray-600/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                      
                      {/* Profile Image */}
                      <div className="relative w-63 h-70 mx-auto mb-6 rounded-lg overflow-hidden border-1 border-gray-500/20 group-hover:border-gray-300/40 transition-all duration-300 transform group-hover:scale-105">
                      <img
                        src={dev.img}
                        alt={`Profile picture of ${dev.name}`}
                        className="rounded-lg w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                      />
                      </div>
  
                      {/* Content */}
                      <div className="text-center space-y-3">
                        <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-600 via-gray-500 to-gray-800">{dev.name}</h3>
                        <p className="text-[16px] text-slate-500/80">{dev.role}</p>
                        
                        {/* Social Links */}
                        <div className="flex justify-center gap-4 mt-4">
                          <a href={dev.linkedin} target="_blank" rel="noopener noreferrer" className="p-2 rounded-full bg-gray-300/10 hover:bg-gray-500/20 transition-all duration-300 hover:scale-110">
                            <FaLinkedin className="w-5 h-5 text-slate-400" />
                          </a>
                          <a href={dev.instagram} target="_blank" rel="noopener noreferrer" className="p-2 rounded-full bg-gray-300/10 hover:bg-gray-500/20 transition-all duration-300 hover:scale-110">
                            <FaInstagram className="w-5 h-5 text-slate-400" />
                          </a>
                        </div>
                      </div>
                    </div>
                  </div>
  
                  {/* Back Side */}
                  <div className="absolute inset-0 h-full w-full [transform:rotateY(180deg)] [backface-visibility:hidden]">
                    <div className="h-full bg-gradient-to-br from-black/90 to-black/95 backdrop-blur-2xl border border-gray-500/20 rounded-2xl p-8 flex items-center justify-center text-center">
                    <div className="absolute inset-0 bg-gradient-to-b from-blue-300/10 via-transparent to-gray-600/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none rounded-2xl"></div>

                      <div className="space-y-6">
                        <p className="text-lg text-slate-100 leading-relaxed">{dev.description}</p>
                        <div className="flex justify-center gap-4">
                          <a href={dev.linkedin} target="_blank" rel="noopener noreferrer" className="p-2 rounded-full bg-gray-300/10 hover:bg-gray-500/20 transition-all duration-300 hover:scale-110">
                            <FaLinkedin className="w-5 h-5 text-slate-400" />
                          </a>
                          <a href={dev.instagram} target="_blank" rel="noopener noreferrer" className="p-2 rounded-full bg-gray-300/10 hover:bg-gray-500/20 transition-all duration-300 hover:scale-110">
                            <FaInstagram className="w-5 h-5 text-slate-400" />
                          </a>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
}
