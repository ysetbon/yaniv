import { useState } from 'react';
import { GameBoard } from './components/GameBoard';
import { StartScreen } from './components/StartScreen';
import './App.css';

function App() {
  const [gameStarted, setGameStarted] = useState(false);
  const [playerNames, setPlayerNames] = useState<string[]>([]);

  const handleStartGame = (names: string[]) => {
    setPlayerNames(names);
    setGameStarted(true);
  };

  const handleEndGame = () => {
    setGameStarted(false);
    setPlayerNames([]);
  };

  return (
    <div className="app">
      {!gameStarted ? (
        <StartScreen onStartGame={handleStartGame} />
      ) : (
        <GameBoard playerNames={playerNames} onEndGame={handleEndGame} />
      )}
    </div>
  );
}

export default App;