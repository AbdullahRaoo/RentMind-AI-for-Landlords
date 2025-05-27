import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Chatbot from "./Chatbot";
import "./Chatbot.css";

function App() {
  const [count, setCount] = useState(0)

  return (
    <>

      <Chatbot />
    </>
  )
}

export default App
