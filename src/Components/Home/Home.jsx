import React from 'react'
import Design from '../../Spline/Design.jsx'
import CurvedLine from '../CurvedLine/CurvedLine.jsx'
import { NavLink } from 'react-router-dom'
import Dashboard from '../Dashboard/Dashboard.jsx'

function Home() {
    return (
        <>
            <section className="h-full flex justify-center items-center bg-black px-6">
            <div className="grid md:grid-cols-2 gap-10 items-center max-w-4xl w-full">
                
                {/* Left Section */}
                <div className="bg-transparent p-6 rounded-lg shadow-lg text-white order-2 md:order-1">
                    <h1 className="text-4xl font-bold mb-4">
                        <span className='bg-gradient-to-r from-pink-700 to-red-500 bg-clip-text text-transparent'>
                            Welcome to Ajax
                        </span>
                        
                    </h1>
                    <p className="text-[14px] font-light text-gray-400">
                        Lorem ipsum dolor sit amet consectetur adipisicing elit. Ullam quibusdam nisi dolorem. 
                        Quibusdam optio vitae laborum reiciendis molestias reprehenderit doloribus ipsam beatae, 
                        possimus dolorum nemo deleniti repellendus ducimus nobis asperiores!
                    </p>
                </div>

                
                <div className="relative flex justify-center items-center pt-17 pb-17 order-1 md:order-2">
                    <div className="h-[380px] w-[300px] relative">
                        <Design />                      
                        <div className="absolute inset-0 bg-gradient-to-r from-black via-transparent to-black pointer-events-none"></div>
                    </div>
                </div>
            </div>
            </section>
            <CurvedLine />
            <div>
                <Dashboard />
            </div>
            
        </>
    )
}

export default Home