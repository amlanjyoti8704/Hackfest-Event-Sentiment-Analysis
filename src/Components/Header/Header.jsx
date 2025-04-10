import React, { useState, useRef, useEffect } from 'react'
import { NavLink, Link } from 'react-router-dom'
import { FiMenu, FiX } from 'react-icons/fi'

function Header() {
    const [isOpen, setIsOpen] = useState(false)
    const menuRef = useRef(null)

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setIsOpen(false)
            }
        }

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside)
        } else {
            document.removeEventListener('mousedown', handleClickOutside)
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside)
        }
    }, [isOpen])
    return (
        <>
        <header className='shadow sticky top-0 z-50 bg-black'>
            <nav className='lg-px-6 px-4 pl-0 py-2.5 pt-0'>
                <div className='flex flex-wrap justify-between items-center mx-auto max-w-screen-xl relative'>
                    <button onClick={()=>setIsOpen(!isOpen)} className='md:hidden text-white text-3xl focus:outline-none order-first' aria-expanded={isOpen}>
                        {isOpen ? <FiX className='absolute ml-4 z-3 -mt-1'/> : <FiMenu className=' ml-4 mt-2.5' />}
                    </button>
                    <div className='pt-2.5 flex-grow flex justify-center md:justify-start'>
                        <Link to='/' className='flex items-center' >
                            <img
                                src="../src/assets/image.png"
                                className="mr-3 h-16 "
                                alt="Logo"
                            />
                        </Link>
                    </div>
                    <div 
                    ref={menuRef}
                    className={`md:flex flex-col md:flex-row md:space-x-10 md:items-center 
                                    absolute md:relative md:right-0 md:ml-auto top-0 left-0 w-[40vw] md:w-auto 
                                    h-screen md:h-auto bg-gray-950 md:bg-transparent p-7 pt-10 md:p-0 
                                    transition-transform duration-300 ease-in-out 
                                    ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}`}>
                            
                        <ul className='flex flex-col mt-10 font-medium md:flex-row md:space-x-10 md:mt-0 md:items-center md:mr-5'>
                            <li>
                                <NavLink to="/" className={({isActive}) => `${isActive?"border-b border-gray-100 text-orange-700":"lg:border-0 text-shadow-gray-400"} block py-2 pr-2 pl-1 duration-200 text-gray-400 hover:bg-transparent  hover:text-orange-700 lg:p-0`}>
                                    Home
                                </NavLink>
                            </li>
                            <li>
                                <NavLink to="/about" className={({isActive}) => `${isActive?"border-b border-gray-100 text-orange-700":"lg:border-0 text-shadow-gray-400"} block py-2 pr-2 pl-1 duration-200 text-gray-400 hover:bg-transparent  hover:text-orange-700 lg:p-0`}>
                                    About
                                </NavLink>
                            </li>
                            <li>
                                <NavLink to='/developers' className={({isActive}) => `${isActive?"border-b border-gray-100 text-orange-700":"lg:border-0 text-shadow-gray-400"} block py-2 pr-2 pl-1 duration-200 text-gray-400 hover:bg-transparent  hover:text-orange-700 lg:p-0`}>
                                    Developers
                                </NavLink>
                            </li>
                            <li>
                                <NavLink to="/play" className={({isActive}) => `${isActive?"border-b border-gray-100 text-orange-700":"lg:border-0 text-shadow-gray-400"} block py-2 pr-2 pl-1 duration-200 text-gray-400 hover:bg-transparent  hover:text-orange-700 lg:p-0`}>
                                    Play
                                </NavLink>
                            </li>
                            <li>
                                <NavLink to="/contactUs" className={({isActive}) => `${isActive?"border-b border-gray-100 text-orange-700":"lg:border-0 text-shadow-gray-400"} block py-2 pr-2 pl-1 duration-200 text-gray-400 hover:bg-transparent  hover:text-orange-700 lg:p-0`}>
                                    Contact Us
                                </NavLink>
                                
                            </li>
                            
                        </ul>
                    </div>
                </div>
            </nav>
        </header>
        </>
    )
}

export default Header