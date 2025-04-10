import React, { useState } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faInstagram, faLinkedin, faTwitter } from "@fortawesome/free-brands-svg-icons";

function Card() {
  const [flipped, setFlipped] = useState(false);

  return (
    <div className="flex justify-center items-center h-[50vh] md:h-[80vh] bg-transparent">
      {/* Card Container with Hover Zoom */}
      <div
        className="relative w-96 md:w-80 h-56 md:h-[65vh] transform transition-transform duration-500 hover:scale-105 cursor-pointer"
        onClick={() => setFlipped(!flipped)}
        onMouseLeave={() => setFlipped(false)} // Reset when cursor leaves
      >
        {/* Inner Card with Flip Effect */}
        <div className={`relative w-full h-full transition-transform duration-500 transform-style-preserve-3d ${flipped ? "rotate-y-180" : ""}`}>
          
          {/* Front Side */}
          <div className="absolute inset-0 shadow-lg rounded-lg overflow-hidden flex flex-col bg-gray-700 transition-transform duration-500 backface-hidden">
            
            {/* Image Section */}
            <div>
              <img 
                className="w-full h-40 md:h-90 object-cover" 
                src="../src/assets/image copy.png" 
                alt="Profile" 
              />
            </div>

            {/* Text & Socials Section */}
            <div className="p-4 bg-gray-800 text-center flex flex-col justify-between h-full">
              <h2 className="text-2xl font-semibold text-white">AMLAN JYOTI</h2>

              {/* Social Media Icons */}
              <div className="flex justify-center space-x-4 mt-4">
                <Link to="https://www.instagram.com/" target="_blank">
                  <FontAwesomeIcon icon={faInstagram} className="text-pink-500 text-2xl hover:text-pink-400 transition" />
                </Link>
                <Link to="https://www.linkedin.com/" target="_blank">
                  <FontAwesomeIcon icon={faLinkedin} className="text-blue-500 text-2xl hover:text-blue-400 transition" />
                </Link>
                <Link to="https://twitter.com/" target="_blank">
                  <FontAwesomeIcon icon={faTwitter} className="text-blue-400 text-2xl hover:text-blue-300 transition" />
                </Link>
              </div>
            </div>
          </div>

          {/* Back Side */}
          <div className="absolute inset-0 shadow-lg rounded-lg overflow-hidden bg-gray-900 flex flex-col justify-center items-center transition-transform duration-500 rotate-y-180 backface-hidden">
            <h2 className="text-xl font-semibold text-white not-rotated">More About Me</h2>
            <p className="text-gray-400 text-sm px-6 mt-2 text-center not-rotated">
              I'm a Software Engineer passionate about Web Development and AI. Love working on exciting projects!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Card;