import { useState } from 'react';

interface StartScreenProps {
  onStartGame: (playerNames: string[]) => void;
}

export function StartScreen({ onStartGame }: StartScreenProps) {
  const [playerName, setPlayerName] = useState('Player');

  const handleStart = () => {
    if (playerName.trim()) {
      onStartGame([playerName.trim(), 'Computer']);
    }
  };

  return (
    <div className="start-screen">
      <h1>Yaniv Card Game</h1>
      <div className="start-form">
        <div className="input-group">
          <label htmlFor="player">Your Name:</label>
          <input
            id="player"
            type="text"
            value={playerName}
            onChange={(e) => setPlayerName(e.target.value)}
            maxLength={20}
            onKeyPress={(e) => e.key === 'Enter' && handleStart()}
          />
        </div>
        <button className="start-button" onClick={handleStart}>
          Play vs Computer
        </button>
      </div>
      <div className="rules-summary">
        <h3>Quick Rules:</h3>
        <ul>
          <li>Get your hand value to 7 or less to call "Yaniv"</li>
          <li>Draw cards, then discard cards of equal value</li>
          <li>Discard singles, sets (same rank), or runs (3+ consecutive same suit)</li>
          <li>If someone has equal or lower score when you call Yaniv, you get 30 penalty points!</li>
          <li>First to 101 points loses</li>
        </ul>
      </div>
    </div>
  );
}