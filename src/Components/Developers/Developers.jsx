import React from 'react';
import Card from '../Card/Card.jsx';

function Developers() {
    return (
        <div className="flex flex-col items-center p-6">
            <h1 className="text-2xl font-semibold mb-6">Developers</h1>
            
            {/* Grid Layout for Cards */}
            <div className="px-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                <Card />
                <Card />
                <Card />
                <Card />
                <Card />
            </div>
        </div>
    );
}

export default Developers;