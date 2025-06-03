import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'  // Original multiplayer version
// import App from './App-AI.tsx'  // AI version
// import App from './App-Simple.tsx'  // Test version
// import App from './App-WithAI.tsx'  // Combined version with menu
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)