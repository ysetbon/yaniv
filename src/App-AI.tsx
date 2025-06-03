import { useState } from 'react';
import GameBoardAI from './components/GameBoardAI';
import './App.css';

function App() {
  return (
    <div className="app">
      <div style={{ textAlign: 'center', padding: '20px' }}>
        <h1>Yaniv Card Game - AI Opponent</h1>
        <p>You vs Neural Network AI (Trained with {1020} episodes)</p>
      </div>
      <GameBoardAI />
    </div>
  );
}

export default App;