import React from 'react'
import { useState, useEffect } from 'react'
import { NavLink } from 'react-router-dom'
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    BarChart,
    Bar,
    PieChart,
    Pie,
    Cell,
  } from 'recharts';

const COLORS = ['#f97316', '#fb923c', '#facc15', '#a3e635', '#4ade80'];
const dummyData = {
    reddit: [
      {name: 'Positive', value: 400},
      {name: 'Negative', value: 300},
      {name: 'Neutral', value: 300},
    ],
    twitter: [
        {name: 'Positive', value: 100},
        {name: 'Negative', value: 200},
        {name: 'Neutral', value: 300},
    ],
    threads: [
        {name: 'Positive', value: 200},
        {name: 'Negative', value: 300},
        {name: 'Neutral', value: 400},
    ],
  };
  const dummyData2 = {
    positive:[
        {name: 'reddit', value: 300},
        {name: 'twitter', value: 400},
        {name: 'threads', value: 300},
    ],
    negative: [
        {name: 'reddit', value: 200},
        {name: 'twitter', value: 100},
        {name: 'threads', value: 300},
    ],
    neutral: [
        {name: 'reddit', value: 300},
        {name: 'twitter', value: 200},
        {name: 'threads', value: 400},
    ],
  };

  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, value, index }) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 1.4; // move label further out
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);
  
    return (
      <text
        x={x}
        y={y}
        fill={COLORS[index % COLORS.length]}
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        fontSize={16}
        fontWeight={600}
      >
        {value}
      </text>
    );
  };
  


function Dashboard() {
    const [currentTab, setCurrentTab] = useState('reddit')
    const [currentSentiment, setCurrentSentiment] = useState('positive')
    const renderAppGraph = () => {
        const data = dummyData2[currentSentiment]
        return(
            <ResponsiveContainer width="100%" height={350} className='mt-10 pb-2 pr-8'>
                {
                   (
                    <PieChart>
                        <Pie
                            data={data}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            outerRadius={120}
                            innerRadius={60}
                            fill="#f97316"
                            label={renderCustomLabel}
                            isAnimationActive={false} // optional: disables animation
                            activeIndex={-1} // disables highlight
                            activeShape={null} // disables that annoying rectangle
                        >
                            {data.map((_, index) => (
                            <Cell
                                key={`cell-${index}`}
                                fill={COLORS[index % COLORS.length]}
                                stroke="#1f2937"
                                strokeWidth={2}
                            />
                            ))}
                        </Pie>
                    </PieChart>
                    ) 
                }
            </ResponsiveContainer>
        )
    }
    const renderGraph = () => {
        const data = dummyData[currentTab]
        return(
            <ResponsiveContainer width="100%" height={350} className='mt-10 pb-2 pr-8'>
            {/* {currentTab === 'reddit' ? (
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#f995" strokeWidth={3} />
              </LineChart>
            ) : (
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#f995" radius={[10, 10, 0, 0]} />
              </BarChart>
            )} */}
            {
                (
                    <BarChart data={data}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      {/* <Tooltip /> */}
                      <Bar dataKey="value" fill="#f995" radius={[10, 10, 0, 0]} />
                    </BarChart>
                )
            }
            </ResponsiveContainer>
        )
    }
    

    return (
        <>
            <div>
                <h1 className='text-center text-5xl font-bold mt-10'>
                    <span className='bg-gradient-to-r from-pink-700 to-red-500 text-transparent bg-clip-text'>
                        Dashboard
                    </span>
                </h1>

                <div className='flex justify-center items-center mt-10'>
                    <select className="bg-gray-900 text-slate-400 p-2 rounded-md border border-gray-600" value={currentSentiment} onChange={(e) => setCurrentSentiment(e.target.value)}>
                        {['positive', 'negative', 'neutral'].map((option) => (
                            <option 
                                key={option} 
                                value={option}
                                
                            >
                                {option.charAt(0).toUpperCase() + option.slice(1)}
                            </option>
                        ))}
                    </select>
                </div>

                <div className='flex justify-center items-center'>
                    <div className='bg-gradient-to-br from-black via-gray-900/50 to-gray-950 rounded-lg  border-gray-700  h-100 w-[80vw] mt-20 mb-20'>
                        {renderAppGraph()}
                    </div>
                </div>

                <div className='flex justify-center items-center mt-1 '>
                    <ul className='flex flex-row mt-2 font-medium  space-x-10 items-center'>
                        {['reddit', 'twitter', 'threads'].map((tab) => (
                            <li key={tab}>
                                <NavLink
                                onClick={() => setCurrentTab(tab)}
                                className={`${
                                    currentTab === tab
                                    ? 'border-b border-gray-100 text-orange-700'
                                    : 'border-0 text-gray-400'
                                } block py-2 pr-2 pl-1 duration-200 text-gray-400 hover:bg-transparent  hover:text-orange-700 lg:p-0`}
                                >
                                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                </NavLink>
                            </li>
                        ))}
                        
                    </ul>
                </div>
                <div className='flex justify-center items-center'>
                    <div className='bg-gradient-to-br from-gray-900 via-slate-950/80 to-gray-950 rounded-lg border border-gray-700  h-100 w-[80vw] mt-20 mb-20'>
                        {renderGraph()}
                    </div>
                </div>
        
                
            </div>
        </>
    )
}

export default Dashboard
