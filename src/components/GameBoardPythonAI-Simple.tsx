import { useState, useEffect } from 'react';
import { YanivGame } from '../game/Game';
import { PythonTrainedAI } from '../game/PythonTrainedAI';
import { Card, GameState } from '../types/game';

export default function GameBoardPythonAISimple() {
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [ai, setAI] = useState<PythonTrainedAI | null>(null);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [message, setMessage] = useState('Initializing...');

  // Initialize game
  useEffect(() => {
    const newGame = new YanivGame(['You', 'Python AI']);
    setGame(newGame);
    setGameState(newGame.getState());
    setMessage('Game initialized');
  }, []);

  // Initialize AI
  useEffect(() => {
    const initAI = async () => {
      try {
        setMessage('Loading Python AI model...');
        const newAI = new PythonTrainedAI();
        await newAI.loadModel('/saved_networks/best_overall_optimized.json');
        setAI(newAI);
        setModelLoaded(true);
        setMessage('Python AI loaded successfully!');
      } catch (error) {
        console.error('Error loading AI:', error);
        setMessage(`Failed to load AI: ${error}`);
      }
    };
    
    initAI();
  }, []);

  // Handle AI turn
  useEffect(() => {
    const handleAITurn = async () => {
      if (!game || !gameState || !ai || !modelLoaded) return;
      
      const currentPlayer = gameState.players[gameState.currentPlayerIndex];
      if (currentPlayer.name !== 'Python AI') return;
      
      setMessage('AI is thinking...');
      
      // Add delay for better UX
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      try {
        const aiPlayerIndex = gameState.players.findIndex(p => p.name === 'Python AI');
        const decision = ai.makeMove(gameState, aiPlayerIndex);
        
        if (decision.action === 'yaniv') {
          setMessage('AI called Yaniv!');
          game.callYaniv(currentPlayer.id);
        } else if (decision.action === 'draw') {
          if (decision.source === 'deck') {
            setMessage('AI drew from deck');
            game.drawFromDeck(currentPlayer.id);
          } else {
            setMessage('AI drew from discard');
            game.drawFromDiscard(currentPlayer.id);
          }
        } else if (decision.action === 'discard' && decision.cards) {
          setMessage(`AI discarded ${decision.cards.length} card(s)`);
          game.discard(currentPlayer.id, decision.cards);
        }
        
        setGameState(game.getState());
      } catch (error) {
        console.error('AI error:', error);
        setMessage(`AI error: ${error}`);
      }
    };
    
    handleAITurn();
  }, [game, gameState, ai, modelLoaded]);

  const handleDrawFromDeck = () => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      game.drawFromDeck(currentPlayer.id);
      setGameState(game.getState());
      setMessage('You drew from deck');
    } catch (error) {
      setMessage(`Error: ${error}`);
    }
  };

  const handleDrawFromDiscard = () => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      game.drawFromDiscard(currentPlayer.id);
      setGameState(game.getState());
      setMessage('You drew from discard');
    } catch (error) {
      setMessage(`Error: ${error}`);
    }
  };

  const handleDiscard = (cards: Card[]) => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      game.discard(currentPlayer.id, cards);
      setGameState(game.getState());
      setMessage(`You discarded ${cards.length} card(s)`);
    } catch (error) {
      setMessage(`Error: ${error}`);
    }
  };

  const handleCallYaniv = () => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      const result = game.callYaniv(currentPlayer.id);
      if (result.success) {
        setMessage('You called Yaniv!');
        setGameState(game.getState());
      } else {
        setMessage('Cannot call Yaniv');
      }
    } catch (error) {
      setMessage(`Error: ${error}`);
    }
  };

  if (!game || !gameState) {
    return <div style={{ padding: '20px' }}>Loading game...</div>;
  }

  const humanPlayer = gameState.players.find(p => p.name === 'You');
  const aiPlayer = gameState.players.find(p => p.name === 'Python AI');
  const currentPlayer = gameState.players[gameState.currentPlayerIndex];
  const isHumanTurn = currentPlayer.name === 'You';

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1>Yaniv vs Python AI</h1>
      
      <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '5px' }}>
        <strong>Status:</strong> {message}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <strong>Current Turn:</strong> {currentPlayer.name} | 
        <strong> Phase:</strong> {gameState.turnPhase} |
        <strong> AI Loaded:</strong> {modelLoaded ? '✓' : '✗'}
      </div>

      {/* AI Player */}
      {aiPlayer && (
        <div style={{ marginBottom: '20px', padding: '10px', border: '1px solid #ccc' }}>
          <h3>Python AI ({aiPlayer.hand.length} cards)</h3>
          <div style={{ display: 'flex', gap: '5px' }}>
            {aiPlayer.hand.map((_, index) => (
              <div key={index} style={{ 
                width: '50px', 
                height: '70px', 
                backgroundColor: '#003d82', 
                border: '1px solid #000',
                borderRadius: '5px'
              }} />
            ))}
          </div>
        </div>
      )}

      {/* Deck and Discard */}
      <div style={{ marginBottom: '20px', display: 'flex', gap: '20px', alignItems: 'center' }}>
        <div>
          <h4>Deck</h4>
          <button 
            onClick={handleDrawFromDeck}
            disabled={!isHumanTurn || gameState.turnPhase !== 'draw'}
            style={{ 
              width: '60px', 
              height: '80px', 
              backgroundColor: '#003d82',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
          >
            Draw
          </button>
        </div>
        
        <div>
          <h4>Discard Pile</h4>
          {gameState.discardPile.length > 0 && (
            <button 
              onClick={handleDrawFromDiscard}
              disabled={!isHumanTurn || gameState.turnPhase !== 'draw'}
              style={{ 
                width: '60px', 
                height: '80px', 
                backgroundColor: 'white',
                border: '1px solid #000',
                borderRadius: '5px',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center'
              }}
            >
              <div>{gameState.discardPile[gameState.discardPile.length - 1].rank}</div>
              <div>{gameState.discardPile[gameState.discardPile.length - 1].suit}</div>
            </button>
          )}
        </div>
      </div>

      {/* Human Player */}
      {humanPlayer && (
        <div style={{ padding: '10px', border: '2px solid #007bff' }}>
          <h3>Your Hand</h3>
          <div style={{ display: 'flex', gap: '5px', marginBottom: '10px' }}>
            {humanPlayer.hand.map((card, index) => (
              <div key={index} style={{ 
                width: '60px', 
                height: '80px', 
                backgroundColor: 'white',
                border: '1px solid #000',
                borderRadius: '5px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                cursor: 'pointer'
              }}>
                <div>{card.rank}</div>
                <div>{card.suit}</div>
              </div>
            ))}
          </div>
          
          <div style={{ display: 'flex', gap: '10px' }}>
            <button 
              onClick={() => handleDiscard([humanPlayer.hand[0]])}
              disabled={!isHumanTurn || gameState.turnPhase !== 'discard' || humanPlayer.hand.length === 0}
            >
              Discard First Card
            </button>
            
            <button 
              onClick={handleCallYaniv}
              disabled={!isHumanTurn || !game.canCallYaniv(humanPlayer.id)}
            >
              Call Yaniv
            </button>
          </div>
        </div>
      )}
    </div>
  );
}