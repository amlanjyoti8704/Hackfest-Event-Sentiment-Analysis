import React from 'react'
import Design from '../../Spline/Design.jsx'

export default function About() {
    return (
        <div className="py-16">
            <div className="container m-auto px-6 text-gray-600 md:px-12 xl:px-6">
                <div className="space-y-6 md:space-y-0 md:flex md:gap-6 lg:items-center lg:gap-30">
                    <div className="h-[380px] w-[300px] relative hidden lg:block">
                        <Design />                      
                        <div className="absolute inset-0 bg-gradient-to-r from-black via-transparent to-black/20 pointer-events-none"></div>
                    </div>
                    <div className="md:7/12 lg:w-6/12">
                        <h2 className="text-3xl text-gray-900 font-bold md:text-4xl">
                            <span className='bg-gradient-to-r from-pink-700 via-yellow-700 to-red-500 text-transparent bg-clip-text'>
                                 Something about Project
                            </span>
                            
                        </h2>
                        <p className="mt-6 text-gray-600">
                            Lorem ipsum dolor, sit amet consectetur adipisicing elit. Eum omnis voluptatem
                            accusantium nemo perspiciatis delectus atque autem! Voluptatum tenetur beatae unde
                            aperiam, repellat expedita consequatur! Officiis id consequatur atque doloremque!
                        </p>
                        <p className="mt-4 text-gray-600">
                            Nobis minus voluptatibus pariatur dignissimos libero quaerat iure expedita at?
                            Asperiores nemo possimus nesciunt dicta veniam aspernatur quam mollitia.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
