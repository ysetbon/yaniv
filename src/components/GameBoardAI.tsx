import React, { useState, useEffect } from 'react';
import { YanivGame } from '../game/Game';
import { EnhancedComputerAI } from '../game/EnhancedComputerAI';
import { Card, GameState } from '../types/game';
import { PlayerHand } from './PlayerHand';
import { DiscardPile } from './DiscardPile';
import { ScoreBoard } from './ScoreBoard';
import { GameControls } from './GameControls';
import { GameLog, LogEntry } from './GameLog';
import './GameBoardAI.css';

export default function GameBoardAI() {
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [ai, setAI] = useState<EnhancedComputerAI | null>(null);
  const [aiMode, setAIMode] = useState<'rule-based' | 'neural-network' | 'hybrid'>('rule-based');
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
    const newGame = new YanivGame(['You', 'AI Opponent']);
    setGame(newGame);
    setGameState(newGame.getState());
  }, []);

  // Initialize AI
  useEffect(() => {
    const initAI = async () => {
      console.log('Initializing AI with mode:', aiMode);
      const newAI = new EnhancedComputerAI(aiMode);
      
      // Wait for AI initialization
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Try to load the trained model
      if (newAI['neuralAI']) {
        try {
          const loaded = await newAI['neuralAI'].loadModel('/models/yaniv-ai-model-final/model.json');
          setModelLoaded(loaded);
          console.log('Trained AI model loaded:', loaded);
        } catch (error) {
          console.error('Error loading model:', error);
          setModelLoaded(false);
        }
      }
      
      setAI(newAI);
      console.log('AI initialized');
    };
    
    initAI();
  }, [aiMode]);

  // Handle AI turn
  useEffect(() => {
    const handleAITurn = async () => {
      if (!game || !gameState || !ai) return;
      
      const currentPlayer = gameState.players[gameState.currentPlayerIndex];
      if (currentPlayer.name !== 'AI Opponent' || gameState.gamePhase !== 'playing') return;
      
      setIsAIThinking(true);
      
      // Add delay for better UX
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      try {
        if (gameState.turnPhase === 'discard') {
          // AI decides what to discard
          const canCallYaniv = game.canCallYaniv(currentPlayer.id);
          const decision = await ai.makeDecision(
            currentPlayer.hand,
            gameState.discardPile,
            canCallYaniv,
            gameState,
            currentPlayer.id
          );
          
          if (decision.action === 'yaniv') {
            const handValue = currentPlayer.hand.reduce((sum, card) => sum + card.value, 0);
            addLogEntry({
              player: 'AI Opponent',
              action: 'Called Yaniv',
              cards: currentPlayer.hand,
              details: `Hand value: ${handValue}`
            });
            game.callYaniv(currentPlayer.id);
          } else if (decision.cardsToDiscard) {
            addLogEntry({
              player: 'AI Opponent',
              action: 'Discarded',
              cards: decision.cardsToDiscard,
              details: `Value: ${decision.cardsToDiscard.reduce((sum, c) => sum + c.value, 0)}`
            });
            game.discard(currentPlayer.id, decision.cardsToDiscard);
          }
        } else if (gameState.turnPhase === 'draw') {
          // AI already decided draw source in previous decision
          const decision = await ai.makeDecision(
            currentPlayer.hand,
            gameState.discardPile,
            false,
            gameState,
            currentPlayer.id
          );
          
          if (decision.drawSource === 'deck') {
            addLogEntry({
              player: 'AI Opponent',
              action: 'Drew from deck'
            });
            game.drawFromDeck(currentPlayer.id);
          } else {
            addLogEntry({
              player: 'AI Opponent',
              action: 'Drew from discard pile',
              details: `AI mode: ${aiMode}`
            });
            game.drawFromDiscard(currentPlayer.id);
          }
        }
        
        setGameState(game.getState());
      } catch (error) {
        console.error('AI turn error:', error);
      }
      
      setIsAIThinking(false);
    };
    
    handleAITurn();
  }, [game, gameState, ai]);

  const handleDiscard = () => {
    if (!game || !gameState || selectedCards.length === 0) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      addLogEntry({
        player: 'You',
        action: 'Discarded',
        cards: selectedCards,
        details: `Value: ${selectedCards.reduce((sum, c) => sum + c.value, 0)}`
      });
      game.discard(currentPlayer.id, selectedCards);
      setSelectedCards([]);
      setGameState(game.getState());
    } catch (error) {
      alert(error.message);
    }
  };

  const handleDraw = (source: 'deck' | 'discard') => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      if (source === 'deck') {
        addLogEntry({
          player: 'You',
          action: 'Drew from deck'
        });
        game.drawFromDeck(currentPlayer.id);
      } else {
        addLogEntry({
          player: 'You',
          action: 'Drew from discard pile'
        });
        game.drawFromDiscard(currentPlayer.id);
      }
      setGameState(game.getState());
    } catch (error) {
      alert(error.message);
    }
  };

  const handleYaniv = () => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      const handValue = currentPlayer.hand.reduce((sum, card) => sum + card.value, 0);
      addLogEntry({
        player: 'You',
        action: 'Called Yaniv',
        cards: currentPlayer.hand,
        details: `Hand value: ${handValue}`
      });
      game.callYaniv(currentPlayer.id);
      setGameState(game.getState());
    } catch (error) {
      alert(error.message);
    }
  };

  const handleNewGame = () => {
    const newGame = new YanivGame(['You', 'AI Opponent']);
    setGame(newGame);
    setGameState(newGame.getState());
    setSelectedCards([]);
  };

  if (!gameState || !game) {
    return (
      <div className="loading">
        <h2>Loading game...</h2>
        <p>Initializing AI opponent...</p>
      </div>
    );
  }

  const humanPlayer = gameState.players.find(p => p.name === 'You');
  const aiPlayer = gameState.players.find(p => p.name === 'AI Opponent');
  const isHumanTurn = gameState.players[gameState.currentPlayerIndex].name === 'You';

  return (
    <div className="game-board">
      <GameLog entries={gameLog} />
      
      <div className="ai-status">
        <h3>AI Mode: {aiMode}</h3>
        {modelLoaded && <span className="model-loaded">✓ Trained Model Loaded</span>}
        {isAIThinking && <span className="ai-thinking">AI is thinking...</span>}
        
        <select value={aiMode} onChange={(e) => setAIMode(e.target.value as any)}>
          <option value="rule-based">Rule-based AI</option>
          <option value="neural-network">Neural Network AI</option>
          <option value="hybrid">Hybrid AI (Recommended)</option>
        </select>
      </div>

      <ScoreBoard players={gameState.players} currentPlayerIndex={gameState.currentPlayerIndex} />
      
      {/* AI Opponent's Hand */}
      {aiPlayer && (
        <div className="opponent-area">
          <h3>AI Opponent</h3>
          <PlayerHand
            player={aiPlayer}
            isCurrentPlayer={!isHumanTurn}
            onCardSelect={() => {}}
            selectedCards={[]}
            showCards={false}
          />
        </div>
      )}

      <DiscardPile cards={gameState.discardPile} />

      {/* Human Player's Hand */}
      {humanPlayer && (
        <div className="player-area">
          <h3>Your Hand</h3>
          <PlayerHand
            player={humanPlayer}
            isCurrentPlayer={isHumanTurn}
            onCardSelect={(card) => {
              if (!isHumanTurn) return;
              
              const isSelected = selectedCards.some(c => 
                c.suit === card.suit && c.rank === card.rank
              );
              
              if (isSelected) {
                setSelectedCards(selectedCards.filter(c => 
                  !(c.suit === card.suit && c.rank === card.rank)
                ));
              } else {
                setSelectedCards([...selectedCards, card]);
              }
            }}
            selectedCards={selectedCards}
            showCards={true}
          />
        </div>
      )}

      {isHumanTurn && (
        <GameControls
          onDiscard={handleDiscard}
          onDrawFromDeck={() => handleDraw('deck')}
          onDrawFromDiscard={() => handleDraw('discard')}
          onCallYaniv={handleYaniv}
          canCallYaniv={humanPlayer ? game.canCallYaniv(humanPlayer.id) : false}
          turnPhase={gameState.turnPhase || 'discard'}
          hasSelectedCards={selectedCards.length > 0}
        />
      )}

      {gameState.gamePhase === 'roundEnd' && (
        <div className="round-end">
          <h2>Round Over!</h2>
          <button onClick={handleNewGame}>New Game</button>
        </div>
      )}

    </div>
  );
}