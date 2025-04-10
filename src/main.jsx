import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import {createBrowserRouter, createRoutesFromElements, Route, RouterProvider} from 'react-router-dom'
import Layout from './Layout.jsx'
import Home from './Components/Home/Home.jsx'
import About from './Components/About/About.jsx'
// import Developers from './Components/Developers/Developers.jsx'
import DevelopersPage from './Components/Developers/DD.jsx'
import ContactUs from './Components/ContactUs/ContactUs.jsx'
import User from './Components/User/User.jsx'
import Play from './Components/Play/Play.jsx'
import Dashboard from './Components/Dashboard/Dashboard.jsx'


const router=createBrowserRouter(
  createRoutesFromElements(
    <Route path='/' element={<Layout />}>
      <Route path='' element={<Home/>}>
        <Route path='dashboard' element={<Dashboard/>}/>
      </Route>
      <Route path='about' element={<About/>}/>
      <Route path='developers' element={<DevelopersPage/>}/>
      <Route path='user/:userid' element={<User/>}/>
      <Route path='contactus' element={<ContactUs/>}/>
      <Route path='play' element={<Play/>}/>
    </Route>
  )
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RouterProvider router={router}/>
  </StrictMode>,
)
