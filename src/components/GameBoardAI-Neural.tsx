import React, { useState, useEffect } from 'react';
import { YanivGame } from '../game/Game';
import { NeuralNetworkAI } from '../game/NeuralNetworkAI';
import { Card, GameState } from '../types/game';
import { PlayerHand } from './PlayerHand';
import { DiscardPile } from './DiscardPile';
import { ScoreBoard } from './ScoreBoard';
import { GameControls } from './GameControls';
import { GameLog, LogEntry } from './GameLog';
import { CardUtils } from '../game/Card';
import './GameBoardAI.css';

export default function GameBoardAINeuralNetwork() {
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [gameLog, setGameLog] = useState<LogEntry[]>([]);
  const [neuralAI, setNeuralAI] = useState<NeuralNetworkAI | null>(null);
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);

  // Helper function to add log entry
  const addLogEntry = (entry: Omit<LogEntry, 'timestamp'>) => {
    const timestamp = new Date().toLocaleTimeString();
    setGameLog(prev => [...prev, { ...entry, timestamp }]);
  };

  // Initialize neural network AI and load model
  useEffect(() => {
    const initAI = async () => {
      const ai = new NeuralNetworkAI();
      
      // Try to load the trained model
      const modelPath = '/models/yaniv-trained-4000/model.json';
      const loaded = await ai.loadModel(modelPath);
      
      setNeuralAI(ai);
      setModelLoaded(loaded);
      
      if (loaded) {
        addLogEntry({
          player: 'System',
          action: 'Neural Network AI loaded successfully'
        });
      } else {
        addLogEntry({
          player: 'System',
          action: 'Failed to load trained model, using default AI'
        });
      }
    };
    
    initAI();
  }, []);

  // Initialize game after AI is ready
  useEffect(() => {
    if (neuralAI !== null) {
      const newGame = new YanivGame(['You', 'AI (Neural Network)']);
      setGame(newGame);
      setGameState(newGame.getState());
      addLogEntry({
        player: 'System',
        action: 'New game started'
      });
    }
  }, [neuralAI]);

  // Handle AI turn
  useEffect(() => {
    if (!game || !gameState || isAIThinking || !neuralAI) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'AI (Neural Network)' || gameState.gamePhase !== 'playing') return;
    
    const makeAIMove = async () => {
      setIsAIThinking(true);
      await new Promise(resolve => setTimeout(resolve, 1000)); // Delay for UX
      
      try {
        // Use neural network to make decision
        const decision = await neuralAI.makeDecision(
          currentPlayer.hand,
          gameState.discardPile,
          gameState,
          currentPlayer.id
        );
        
        if (gameState.turnPhase === 'discard') {
          if (decision.action === 'yaniv' && game.canCallYaniv(currentPlayer.id)) {
            const handValue = CardUtils.getHandValue(currentPlayer.hand);
            addLogEntry({
              player: 'AI (Neural Network)',
              action: 'Called Yaniv',
              cards: currentPlayer.hand,
              details: `Hand value: ${handValue}`
            });
            game.callYaniv(currentPlayer.id);
          } else if (decision.action === 'discard' && decision.cardsToDiscard) {
            addLogEntry({
              player: 'AI (Neural Network)',
              action: 'Discarded',
              cards: decision.cardsToDiscard,
              details: `Value: ${CardUtils.getHandValue(decision.cardsToDiscard)}`
            });
            const success = game.discard(currentPlayer.id, decision.cardsToDiscard);
            
            if (!success) {
              // Fallback: discard first card
              const cardToDiscard = [currentPlayer.hand[0]];
              addLogEntry({
                player: 'AI (Neural Network)',
                action: 'Discarded (fallback)',
                cards: cardToDiscard,
                details: `Value: ${cardToDiscard[0].value}`
              });
              game.discard(currentPlayer.id, cardToDiscard);
            }
          } else {
            // Fallback: discard first card
            const cardToDiscard = [currentPlayer.hand[0]];
            addLogEntry({
              player: 'AI (Neural Network)',
              action: 'Discarded (fallback)',
              cards: cardToDiscard,
              details: `Value: ${cardToDiscard[0].value}`
            });
            game.discard(currentPlayer.id, cardToDiscard);
          }
        } else if (gameState.turnPhase === 'draw') {
          if (decision.action === 'draw') {
            if (decision.drawSource === 'discard' && gameState.discardPile.length > 0) {
              const drawCount = decision.drawCount || 1;
              addLogEntry({
                player: 'AI (Neural Network)',
                action: `Drew ${drawCount} from discard pile`
              });
              game.drawFromDiscard(currentPlayer.id, drawCount);
            } else {
              addLogEntry({
                player: 'AI (Neural Network)',
                action: 'Drew from deck'
              });
              game.drawFromDeck(currentPlayer.id);
            }
          } else {
            // Fallback: draw from deck
            addLogEntry({
              player: 'AI (Neural Network)',
              action: 'Drew from deck (fallback)'
            });
            game.drawFromDeck(currentPlayer.id);
          }
        }
        
        setGameState(game.getState());
      } catch (error) {
        console.error('AI move error:', error);
        addLogEntry({
          player: 'System',
          action: 'AI error occurred, using fallback move'
        });
        
        // Fallback moves
        if (gameState.turnPhase === 'discard') {
          game.discard(currentPlayer.id, [currentPlayer.hand[0]]);
        } else {
          game.drawFromDeck(currentPlayer.id);
        }
        setGameState(game.getState());
      } finally {
        setIsAIThinking(false);
      }
    };
    
    makeAIMove();
  }, [game, gameState, isAIThinking, neuralAI]);

  const handleCardSelect = (card: Card) => {
    if (!gameState || gameState.currentPlayerIndex !== 0 || gameState.turnPhase !== 'discard') return;
    
    setSelectedCards(prev => {
      const isSelected = prev.some(c => c.suit === card.suit && c.rank === card.rank);
      if (isSelected) {
        return prev.filter(c => !(c.suit === card.suit && c.rank === card.rank));
      } else {
        return [...prev, card];
      }
    });
  };

  const handleDiscard = () => {
    if (!game || !gameState || selectedCards.length === 0) return;
    
    const success = game.discard(gameState.players[0].id, selectedCards);
    if (success) {
      addLogEntry({
        player: 'You',
        action: 'Discarded',
        cards: selectedCards,
        details: `Value: ${CardUtils.getHandValue(selectedCards)}`
      });
      setSelectedCards([]);
      setGameState(game.getState());
    }
  };

  const handleDrawFromDeck = () => {
    if (!game || !gameState) return;
    
    game.drawFromDeck(gameState.players[0].id);
    addLogEntry({
      player: 'You',
      action: 'Drew from deck'
    });
    setGameState(game.getState());
  };

  const handleDrawFromDiscard = () => {
    if (!game || !gameState) return;
    
    game.drawFromDiscard(gameState.players[0].id);
    addLogEntry({
      player: 'You',
      action: 'Drew from discard pile'
    });
    setGameState(game.getState());
  };

  const handleCallYaniv = () => {
    if (!game || !gameState) return;
    
    const result = game.callYaniv(gameState.players[0].id);
    const handValue = CardUtils.getHandValue(gameState.players[0].hand);
    
    if (result.success) {
      addLogEntry({
        player: 'You',
        action: 'Called Yaniv',
        cards: gameState.players[0].hand,
        details: `Hand value: ${handValue}${result.assaf ? ' - ASSAF!' : ''}`
      });
    }
    setGameState(game.getState());
  };

  const handleNewGame = () => {
    if (!game) return;
    
    game.resetGame();
    setGameState(game.getState());
    setSelectedCards([]);
    setGameLog([]);
    addLogEntry({
      player: 'System',
      action: 'New game started'
    });
  };

  if (!gameState || !neuralAI) {
    return <div className="loading">Loading Neural Network AI...</div>;
  }

  const currentPlayer = gameState.players[gameState.currentPlayerIndex];
  const isPlayerTurn = gameState.currentPlayerIndex === 0;
  const canCallYaniv = isPlayerTurn && game && game.canCallYaniv(currentPlayer.id);

  return (
    <div className="game-board-ai">
      <div className="main-content">
        <h2 className="game-title">Yaniv - Playing vs AI (Neural Network)</h2>
        {modelLoaded === false && (
          <div className="warning">Failed to load trained model, using default AI</div>
        )}
        
        <ScoreBoard players={gameState.players} gamePhase={gameState.gamePhase} />
        
        {gameState.gamePhase === 'ended' && (
          <div className="game-over">
            <h2>Game Over!</h2>
            <p>Winner: {gameState.players.find(p => p.score < 101)?.name}</p>
            <button onClick={handleNewGame}>New Game</button>
          </div>
        )}
        
        <div className="opponent-area">
          <h3>AI (Neural Network)</h3>
          <PlayerHand 
            cards={Array(gameState.players[1].hand.length).fill(null)} 
            isCurrentPlayer={false}
            onCardSelect={() => {}}
            selectedCards={[]}
          />
          {isAIThinking && <div className="thinking">AI is thinking...</div>}
        </div>
        
        <DiscardPile cards={gameState.discardPile} />
        
        <div className="player-area">
          <h3>Your Hand</h3>
          <PlayerHand 
            cards={currentPlayer.hand}
            isCurrentPlayer={isPlayerTurn}
            onCardSelect={handleCardSelect}
            selectedCards={selectedCards}
          />
        </div>
        
        <GameControls
          isCurrentPlayer={isPlayerTurn}
          turnPhase={gameState.turnPhase}
          selectedCards={selectedCards}
          canCallYaniv={canCallYaniv}
          onDiscard={handleDiscard}
          onDrawFromDeck={handleDrawFromDeck}
          onDrawFromDiscard={handleDrawFromDiscard}
          onCallYaniv={handleCallYaniv}
          discardPileEmpty={gameState.discardPile.length === 0}
        />
        
        <div className="turn-indicator">
          Current Turn: {currentPlayer.name}
        </div>
      </div>
      
      <GameLog entries={gameLog} />
    </div>
  );
}