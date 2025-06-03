import React, { useState, useEffect } from 'react';
import { YanivGame } from '../game/Game';
import { PythonTrainedAI } from '../game/PythonTrainedAI';
import { Card, GameState } from '../types/game';
import { PlayerHand } from './PlayerHand';
import { DiscardPile } from './DiscardPile';
import { ScoreBoard } from './ScoreBoard';
import { GameControls } from './GameControls';
import { GameLog, LogEntry } from './GameLog';
import './GameBoardAI.css';

export default function GameBoardPythonAI() {
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [ai, setAI] = useState<PythonTrainedAI | null>(null);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [gameLog, setGameLog] = useState<LogEntry[]>([]);

  // Helper function to add log entry
  const addLogEntry = (entry: Omit<LogEntry, 'timestamp'>) => {
    const timestamp = new Date().toLocaleTimeString();
    setGameLog(prev => [...prev, { ...entry, timestamp }]);
  };

  // Initialize game
  useEffect(() => {
    const newGame = new YanivGame(['You', 'Python AI']);
    setGame(newGame);
    setGameState(newGame.getState());
  }, []);

  // Initialize AI
  useEffect(() => {
    const initAI = async () => {
      console.log('Initializing Python-trained AI...');
      const newAI = new PythonTrainedAI();
      
      try {
        await newAI.loadModel('/saved_networks/best_overall_optimized.json');
        setModelLoaded(true);
        setAI(newAI);
        console.log('Python AI model loaded successfully');
        addLogEntry({
          player: 'System',
          action: 'AI Loaded',
          details: 'Python-trained neural network loaded'
        });
      } catch (error) {
        console.error('Error loading Python AI model:', error);
        setModelLoaded(false);
        addLogEntry({
          player: 'System',
          action: 'AI Error',
          details: 'Failed to load Python AI model'
        });
      }
    };
    
    initAI();
  }, []);

  // Handle AI turn
  useEffect(() => {
    const handleAITurn = async () => {
      if (!game || !gameState || !ai || !modelLoaded) return;
      
      const currentPlayer = gameState.players[gameState.currentPlayerIndex];
      if (currentPlayer.name !== 'Python AI' || gameState.gamePhase !== 'playing') return;
      
      setIsAIThinking(true);
      
      // Add delay for better UX
      await new Promise(resolve => setTimeout(resolve, 800));
      
      try {
        const aiPlayerIndex = gameState.players.findIndex(p => p.name === 'Python AI');
        const decision = ai.makeMove(gameState, aiPlayerIndex);
        
        if (decision.action === 'yaniv') {
          const handValue = currentPlayer.hand.reduce((sum, card) => sum + card.value, 0);
          addLogEntry({
            player: 'Python AI',
            action: 'Called Yaniv',
            cards: currentPlayer.hand,
            details: `Hand value: ${handValue}`
          });
          game.callYaniv(currentPlayer.id);
        } else if (decision.action === 'draw') {
          const drawnCards = game.drawCards(currentPlayer.id, decision.source || 'deck');
          addLogEntry({
            player: 'Python AI',
            action: `Drew from ${decision.source || 'deck'}`,
            cards: drawnCards
          });
        } else if (decision.action === 'discard' && decision.cards) {
          addLogEntry({
            player: 'Python AI',
            action: 'Discarded',
            cards: decision.cards
          });
          game.discardCards(currentPlayer.id, decision.cards);
        }
        
        setGameState(game.getState());
      } catch (error) {
        console.error('AI error:', error);
        addLogEntry({
          player: 'Python AI',
          action: 'Error',
          details: 'AI made an invalid move'
        });
      } finally {
        setIsAIThinking(false);
      }
    };
    
    handleAITurn();
  }, [game, gameState, ai, modelLoaded]);

  const handleDrawCard = (source: 'deck' | 'discard') => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    const drawnCards = game.drawCards(currentPlayer.id, source);
    addLogEntry({
      player: 'You',
      action: `Drew from ${source}`,
      cards: drawnCards
    });
    
    setGameState(game.getState());
    setSelectedCards([]);
  };

  const handleDiscardCards = () => {
    if (!game || !gameState || selectedCards.length === 0) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    if (game.canDiscardCards(selectedCards)) {
      addLogEntry({
        player: 'You',
        action: 'Discarded',
        cards: selectedCards
      });
      game.discardCards(currentPlayer.id, selectedCards);
      setGameState(game.getState());
      setSelectedCards([]);
    }
  };

  const handleCallYaniv = () => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    if (game.canCallYaniv(currentPlayer.id)) {
      const handValue = currentPlayer.hand.reduce((sum, card) => sum + card.value, 0);
      addLogEntry({
        player: 'You',
        action: 'Called Yaniv',
        cards: currentPlayer.hand,
        details: `Hand value: ${handValue}`
      });
      game.callYaniv(currentPlayer.id);
      setGameState(game.getState());
    }
  };

  const handleNewRound = () => {
    if (game) {
      game.newRound();
      setGameState(game.getState());
      setSelectedCards([]);
      setGameLog([]);
      addLogEntry({
        player: 'System',
        action: 'New Round',
        details: 'Starting new round'
      });
    }
  };

  const handleNewGame = () => {
    const newGame = new YanivGame(['You', 'Python AI']);
    setGame(newGame);
    setGameState(newGame.getState());
    setSelectedCards([]);
    setGameLog([]);
    addLogEntry({
      player: 'System',
      action: 'New Game',
      details: 'Starting new game'
    });
  };

  const toggleCardSelection = (card: Card) => {
    setSelectedCards(prev => {
      const isSelected = prev.some(c => c.id === card.id);
      if (isSelected) {
        return prev.filter(c => c.id !== card.id);
      } else {
        return [...prev, card];
      }
    });
  };

  if (!game || !gameState) {
    return <div>Loading...</div>;
  }

  const humanPlayer = gameState.players.find(p => p.name === 'You');
  const aiPlayer = gameState.players.find(p => p.name === 'Python AI');
  const currentPlayer = gameState.players[gameState.currentPlayerIndex];
  const isHumanTurn = currentPlayer.name === 'You';

  return (
    <div className="game-board">
      <div className="game-header">
        <h1>Yaniv vs Python AI</h1>
        <div className="ai-status">
          {modelLoaded ? (
            <span className="status-loaded">✓ Python AI Loaded</span>
          ) : (
            <span className="status-loading">Loading AI...</span>
          )}
        </div>
      </div>

      <div className="game-main">
        <div className="opponent-section">
          {aiPlayer && (
            <div className="opponent-hand">
              <h3>Python AI {isAIThinking && '(thinking...)'}</h3>
              <div className="card-backs">
                {aiPlayer.hand.map((_, index) => (
                  <div key={index} className="card card-back" />
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="table-section">
          <div className="deck-area">
            <div 
              className={`deck ${isHumanTurn && gameState.turnPhase === 'draw' ? 'clickable' : ''}`}
              onClick={() => isHumanTurn && gameState.turnPhase === 'draw' && handleDrawCard('deck')}
            >
              <div className="card card-back" />
              <span>{gameState.deck.length} cards</span>
            </div>
          </div>

          <DiscardPile 
            opponentCard={gameState.discardPile.length > 0 ? gameState.discardPile[gameState.discardPile.length - 1] : null}
            totalCards={gameState.discardPile.length}
            onDrawCard={() => handleDrawCard('discard')}
            canDraw={isHumanTurn && gameState.turnPhase === 'draw'}
            isDrawPhase={gameState.turnPhase === 'draw'}
            isHumanTurn={isHumanTurn}
          />

          <div className="game-info">
            <p>Current Turn: <strong>{currentPlayer.name}</strong></p>
            <p>Phase: <strong>{gameState.turnPhase}</strong></p>
            {gameState.gamePhase === 'roundEnd' && (
              <div className="round-end-message">
                <p>{gameState.roundWinner ? `${gameState.roundWinner} wins the round!` : 'Round ended!'}</p>
              </div>
            )}
          </div>
        </div>

        {humanPlayer && (
          <div className="player-section">
            <PlayerHand
              player={humanPlayer}
              isCurrentTurn={isHumanTurn}
              selectedCards={selectedCards}
              onCardSelect={toggleCardSelection}
              drawnCards={[]}
              showCards={true}
            />
            
            <GameControls
              currentPlayer={humanPlayer}
              canDiscard={isHumanTurn && gameState.turnPhase === 'discard' && game.canDiscardCards(selectedCards)}
              canCallYaniv={isHumanTurn && game.canCallYaniv(humanPlayer.id)}
              onDiscard={handleDiscardCards}
              onCallYaniv={handleCallYaniv}
              turnPhase={gameState.turnPhase}
            />
            
            {gameState.gamePhase === 'roundEnd' && (
              <div className="round-controls">
                <button onClick={handleNewRound}>New Round</button>
                <button onClick={handleNewGame}>New Game</button>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="game-sidebar">
        <ScoreBoard 
          players={gameState.players}
          currentPlayerId={currentPlayer.id}
        />
        
        <GameLog entries={gameLog} />
      </div>
    </div>
  );
}