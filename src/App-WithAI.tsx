import { useState } from 'react';
import { GameBoard } from './components/GameBoard';
// import GameBoardAI from './components/GameBoardAI';
// import GameBoardAI from './components/GameBoardAI-Simple';
import GameBoardAI from './components/GameBoardAI-Neural';
import './App.css';

function App() {
  const [gameMode, setGameMode] = useState<'menu' | 'ai' | 'multiplayer'>('menu');
  const [playerNames, setPlayerNames] = useState<string[]>([]);

  const handleStartAIGame = () => {
    setGameMode('ai');
  };

  const handleStartMultiplayer = () => {
    // For simplicity, just start with two players
    setPlayerNames(['Player 1', 'Player 2']);
    setGameMode('multiplayer');
  };

  const handleBackToMenu = () => {
    setGameMode('menu');
    setPlayerNames([]);
  };

  if (gameMode === 'menu') {
    return (
      <div className="app">
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <h1>Yaniv Card Game</h1>
          <div style={{ marginTop: '30px' }}>
            <button 
              onClick={handleStartAIGame}
              style={{
                padding: '15px 30px',
                fontSize: '18px',
                margin: '10px',
                cursor: 'pointer',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '5px'
              }}
            >
              Play vs AI (Neural Network)
            </button>
            <br />
            <button 
              onClick={handleStartMultiplayer}
              style={{
                padding: '15px 30px',
                fontSize: '18px',
                margin: '10px',
                cursor: 'pointer',
                backgroundColor: '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: '5px'
              }}
            >
              Play vs Human
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (gameMode === 'ai') {
    return (
      <div className="app">
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <h1>Yaniv Card Game - AI Opponent</h1>
          <button onClick={handleBackToMenu} style={{ marginBottom: '20px' }}>
            Back to Menu
          </button>
        </div>
        <GameBoardAI />
      </div>
    );
  }

  return (
    <div className="app">
      <GameBoard playerNames={playerNames} onEndGame={handleBackToMenu} />
    </div>
  );
}

export default App;