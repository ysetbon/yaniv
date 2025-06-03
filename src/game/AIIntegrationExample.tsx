import React, { useState, useEffect } from 'react';
import { EnhancedComputerAI, AIMode } from './game/EnhancedComputerAI';
import { trainAI } from './game/trainAI';

// Example component showing how to integrate the neural network AI
export function AIIntegrationExample() {
  const [aiMode, setAIMode] = useState<AIMode>('hybrid');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  
  // Initialize AI
  const [computerAI] = useState(() => new EnhancedComputerAI(aiMode));
  
  useEffect(() => {
    computerAI.setMode(aiMode);
  }, [aiMode, computerAI]);
  
  // Training function
  const handleTrainAI = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    
    try {
      // Train with fewer episodes for demo
      await trainAI(100, 10);
      setTrainingProgress(100);
      alert('Training complete! The AI has been improved.');
    } catch (error) {
      console.error('Training failed:', error);
      alert('Training failed. Check console for details.');
    } finally {
      setIsTraining(false);
    }
  };
  
  // Example of using the AI in your game logic
  const makeAIMove = async (gameState: any) => {
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    const canCallYaniv = gameState.handValue <= 7;
    
    try {
      const decision = await computerAI.makeDecision(
        currentPlayer.hand,
        gameState.discardPile,
        canCallYaniv,
        gameState,
        currentPlayer.id
      );
      
      // Apply the decision to your game state
      console.log('AI Decision:', decision);
      return decision;
    } catch (error) {
      console.error('AI decision failed:', error);
      return null;
    }
  };
  
  return (
    <div style={{ padding: '20px' }}>
      <h2>Neural Network AI Settings</h2>
      
      <div style={{ marginBottom: '20px' }}>
        <label>
          AI Mode:
          <select 
            value={aiMode} 
            onChange={(e) => setAIMode(e.target.value as AIMode)}
            style={{ marginLeft: '10px' }}
          >
            <option value="rule-based">Rule-based (Original)</option>
            <option value="neural-network">Neural Network</option>
            <option value="hybrid">Hybrid (Recommended)</option>
          </select>
        </label>
      </div>
      
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={handleTrainAI}
          disabled={isTraining}
          style={{ 
            padding: '10px 20px',
            fontSize: '16px',
            cursor: isTraining ? 'not-allowed' : 'pointer'
          }}
        >
          {isTraining ? `Training... ${trainingProgress}%` : 'Train AI'}
        </button>
        <p style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
          Training will run self-play games to improve the AI's decision making.
          This may take a few minutes.
        </p>
      </div>
      
      <div style={{ marginTop: '30px' }}>
        <h3>How it works:</h3>
        <ul>
          <li><strong>Rule-based:</strong> Uses the original hand-crafted strategy</li>
          <li><strong>Neural Network:</strong> Uses a trained deep learning model</li>
          <li><strong>Hybrid:</strong> Combines both approaches for best performance</li>
        </ul>
        
        <h3>Training:</h3>
        <p>
          The neural network learns by playing against itself thousands of times.
          It uses a generative approach to:
        </p>
        <ul>
          <li>Encode game states into numerical representations</li>
          <li>Generate action probabilities based on the current situation</li>
          <li>Learn from the outcomes of its decisions</li>
          <li>Improve its strategy over time through reinforcement learning</li>
        </ul>
      </div>
    </div>
  );
}

// Usage in your game component:
/*
// In your game component:
import { EnhancedComputerAI } from './game/EnhancedComputerAI';

const Game = () => {
  const [ai] = useState(() => new EnhancedComputerAI('hybrid'));
  
  const handleComputerTurn = async () => {
    const decision = await ai.makeDecision(
      currentPlayer.hand,
      discardPile,
      canCallYaniv,
      gameState,
      currentPlayer.id
    );
    
    // Apply decision to game
    if (decision.action === 'yaniv') {
      handleYaniv();
    } else if (decision.action === 'draw') {
      if (decision.drawSource === 'deck') {
        drawFromDeck();
      } else {
        drawFromDiscard(decision.drawCount);
      }
    } else if (decision.action === 'discard') {
      discardCards(decision.cardsToDiscard);
    }
  };
};
*/