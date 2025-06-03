import React from 'react'
import ReactDOM from 'react-dom/client'
// import App from './App-Test.tsx'  // Test version
import App from './App-AI.tsx'  // AI version with mode switching
// import App from './App-Simple.tsx'  // Test version
// import App from './App-WithAI.tsx'  // Combined version with menu
// import App from './App-PythonAI.tsx'  // Python-trained AI version (needs fixing)
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)