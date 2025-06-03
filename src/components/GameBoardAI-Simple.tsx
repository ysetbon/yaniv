import React, { useState, useEffect } from 'react';
import { YanivGame } from '../game/Game';
import { Card, GameState } from '../types/game';

export default function GameBoardAISimple() {
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    try {
      console.log('Creating new game...');
      const newGame = new YanivGame(['You', 'AI Opponent']);
      console.log('Game created:', newGame);
      
      setGame(newGame);
      const state = newGame.getState();
      console.log('Game state:', state);
      setGameState(state);
    } catch (err) {
      console.error('Error creating game:', err);
      setError(err.message || 'Failed to create game');
    }
  }, []);

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!gameState) {
    return <div>Loading game state...</div>;
  }

  return (
    <div>
      <h2>Game Loaded Successfully!</h2>
      <p>Players: {gameState.players.map(p => p.name).join(', ')}</p>
      <p>Current player: {gameState.players[gameState.currentPlayerIndex].name}</p>
      <p>Game phase: {gameState.gamePhase}</p>
      
      <h3>Your Hand:</h3>
      <div>
        {gameState.players[0].hand.map((card, i) => (
          <span key={i} style={{ margin: '5px' }}>
            {card.rank}{card.suit}
          </span>
        ))}
      </div>
    </div>
  );
}