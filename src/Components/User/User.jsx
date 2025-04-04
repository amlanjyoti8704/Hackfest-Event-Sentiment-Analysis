import React from 'react'
import { useParams } from 'react-router-dom'

function User() {
    const { userid } = useParams()
    return (
        <div className='text-center font-serif p-[20%] text-white bg-gray-600 w-full h-[60vh]'>User: {userid}</div>
    )
}

export default User