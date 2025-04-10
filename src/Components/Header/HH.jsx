import React, { useState } from 'react' // ✅ Imported useState for managing the mobile menu state
import { NavLink, Link } from 'react-router-dom'
import { FiMenu, FiX } from 'react-icons/fi' // ✅ Imported icons for hamburger menu

function Header() {
    const [isOpen, setIsOpen] = useState(false); // ✅ State to toggle the mobile menu

    return (
        <>
        <header className='shadow sticky top-0 z-10 bg-black'>
            <nav className='lg:px-6 px-4 py-2.5'>
                <div className='flex flex-wrap justify-between items-center mx-auto max-w-screen-xl relative'>

                    {/* ✅ Hamburger Menu Button (Moved to Left) */}
                    <button 
                        onClick={() => setIsOpen(!isOpen)} 
                        className="md:hidden text-white text-3xl focus:outline-none order-first"
                        aria-expanded={isOpen}
                    >
                        {isOpen ? <FiX /> : <FiMenu />} {/* ✅ Shows 'X' when menu is open, '☰' when closed */}
                    </button>

                    {/* ✅ Logo Centered in Mobile View */}
                    <div className="flex-grow flex justify-center md:justify-start">
                        <Link to='/' className='flex items-center'>
                            <img
                                src="../src/assets/image.png"
                                className="h-16"
                                alt="Logo"
                            />
                        </Link>
                    </div>

                    {/* ✅ Navigation Menu */}
                    <div className={`md:flex flex-col md:flex-row md:space-x-10 md:items-center 
                        absolute md:static top-16 left-0 w-full md:w-auto bg-black md:bg-transparent p-5 md:p-0 
                        transition-all duration-300 ${isOpen ? 'block' : 'hidden'}`} // ✅ Mobile menu animation
                    >
                        <ul className='flex flex-col md:flex-row md:space-x-10 md:mt-0 md:items-center'>
                            
                            {/* ✅ Used map() to dynamically generate menu items */}
                            {['/', '/about', '/developers', '/play', '/contactUs'].map((path, index) => (
                                <li key={index}>
                                    <NavLink 
                                        to={path} 
                                        className={({ isActive }) => `${isActive ? "border-b border-gray-100 text-orange-700" : "lg:border-0 text-shadow-gray-400"} 
                                            block py-2 pr-2 pl-1 duration-200 text-gray-400 hover:bg-transparent hover:text-orange-700 lg:p-0`
                                        }
                                    >
                                        {path === '/' ? 'Home' : path.replace('/', '').charAt(0).toUpperCase() + path.slice(2)}
                                    </NavLink>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </nav>
        </header>
        </>
    )
}

export default Header