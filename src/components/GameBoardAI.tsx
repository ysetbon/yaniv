import React, { useState, useEffect } from 'react';
import { YanivGame } from '../game/Game';
import { EnhancedComputerAI } from '../game/EnhancedComputerAI';
import { Card, GameState } from '../types/game';
import { CardUtils } from '../game/Card';
import { PlayerHand } from './PlayerHand';
import { DiscardPile } from './DiscardPile';
import { ScoreBoard } from './ScoreBoard';
import { GameControls } from './GameControls';
import { GameLog, LogEntry } from './GameLog';
import './GameBoardAI.css';

export default function GameBoardAI() {
  console.log('GameBoardAI component rendering');
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [ai, setAI] = useState<EnhancedComputerAI | null>(null);
  const [aiMode, setAIMode] = useState<'rule-based' | 'neural-network' | 'python-trained' | 'hybrid'>('neural-network');
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
    console.log('Initializing game...');
    try {
      const newGame = new YanivGame(['You', 'AI Opponent']);
      console.log('Game created:', newGame);
      setGame(newGame);
      const state = newGame.getState();
      console.log('Game state:', state);
      setGameState(state);
    } catch (error) {
      console.error('Error initializing game:', error);
    }
  }, []);

  // Initialize AI
  useEffect(() => {
    const initAI = async () => {
      console.log('Initializing AI with mode:', aiMode);
      const newAI = new EnhancedComputerAI(aiMode);
      
      // Wait for AI initialization to complete
      await newAI.waitForInit();
      
      // Check if AI is ready
      const isReady = newAI.isReady();
      setModelLoaded(isReady);
      
      if (isReady) {
        console.log(`${aiMode} AI initialized successfully`);
        
        let modeDetails = '';
        switch (aiMode) {
          case 'python-trained':
            modeDetails = '🎯 Using Python-trained AI (Your Model)';
            break;
          case 'neural-network':
            modeDetails = '🧠 Using Neural Network AI (TensorFlow.js)';
            break;
          case 'rule-based':
            modeDetails = '📋 Using Rule-based AI';
            break;
          case 'hybrid':
            modeDetails = '🔀 Using Hybrid AI (Neural + Rules)';
            break;
        }
        
        addLogEntry({
          player: 'System',
          action: 'AI Mode',
          details: modeDetails
        });
      } else {
        console.error(`Failed to initialize ${aiMode} AI`);
        addLogEntry({
          player: 'System',
          action: 'AI Error',
          details: `Failed to load ${aiMode} AI model`
        });
      }
      
      setAI(newAI);
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
      alert(error instanceof Error ? error.message : 'An error occurred');
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
      alert(error instanceof Error ? error.message : 'An error occurred');
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
      alert(error instanceof Error ? error.message : 'An error occurred');
    }
  };

  const handleNewGame = () => {
    const newGame = new YanivGame(['You', 'AI Opponent']);
    setGame(newGame);
    setGameState(newGame.getState());
    setSelectedCards([]);
  };

  if (!gameState || !game) {
    console.log('Game not initialized yet:', { game, gameState });
    return (
      <div style={{ padding: '20px', background: 'white', color: 'black', minHeight: '200px' }}>
        <h2>Loading game...</h2>
        <p>Initializing AI opponent...</p>
        <p>Game: {game ? 'loaded' : 'not loaded'}</p>
        <p>GameState: {gameState ? 'loaded' : 'not loaded'}</p>
      </div>
    );
  }

  const humanPlayer = gameState.players.find(p => p.name === 'You');
  const aiPlayer = gameState.players.find(p => p.name === 'AI Opponent');
  const isHumanTurn = gameState.players[gameState.currentPlayerIndex].name === 'You';

  return (
    <div className="game-board">
      <div className="game-header">
        <h1>Yaniv Card Game</h1>
        <div className="ai-mode-selector">
          <label>AI Mode:</label>
          <select 
            value={aiMode} 
            onChange={(e) => setAIMode(e.target.value as any)}
            className="ai-select"
          >
            <option value="rule-based">Rule-based AI</option>
            <option value="neural-network">Neural Network AI (TensorFlow.js)</option>
            <option value="python-trained">🎯 Python-trained AI (Your Model)</option>
            <option value="hybrid">Hybrid AI (Recommended)</option>
          </select>
          {modelLoaded && <span className="model-status">✓ Ready</span>}
          {aiMode === 'python-trained' && !modelLoaded && (
            <span className="model-loading">Loading...</span>
          )}
        </div>
      </div>

      <div className="game-layout">
        <div className="left-panel">
          <ScoreBoard players={gameState.players} currentPlayerId={gameState.players[gameState.currentPlayerIndex].id} />
          <GameLog entries={gameLog} />
        </div>

        <div className="game-table">
          {/* AI Opponent's Hand at the top */}
          {aiPlayer && (
            <div className="ai-opponent-area">
              <div className="player-label">
                <h3>{aiPlayer.name}</h3>
                {isAIThinking && <span className="thinking-indicator">Thinking...</span>}
              </div>
              <PlayerHand
                player={aiPlayer}
                isCurrentTurn={!isHumanTurn}
                onCardSelect={() => {}}
                selectedCards={[]}
                drawnCards={[]}
                showCards={false}
              />
            </div>
          )}

          {/* Center area with deck and discard pile */}
          <div className="table-center">
            <div className="deck-container">
              <div 
                className={`deck ${isHumanTurn && gameState.turnPhase === 'draw' ? 'drawable' : ''}`}
                onClick={() => isHumanTurn && gameState.turnPhase === 'draw' && handleDraw('deck')}
              >
                <div className="card-stack">
                  <div className="card card-back"></div>
                  <div className="card card-back"></div>
                  <div className="card card-back"></div>
                </div>
                <span className="deck-count">{gameState.deck.length} cards</span>
              </div>
            </div>

            <DiscardPile 
              opponentCard={gameState.discardPile.length > 0 ? gameState.discardPile[gameState.discardPile.length - 1] : null}
              totalCards={gameState.discardPile.length}
              onDrawCard={() => handleDraw('discard')}
              canDraw={isHumanTurn && gameState.turnPhase === 'draw'}
              isDrawPhase={gameState.turnPhase === 'draw'}
              isHumanTurn={isHumanTurn}
            />
          </div>

          {/* Human Player's Hand at the bottom */}
          {humanPlayer && (
            <div className="human-player-area">
              <PlayerHand
                player={humanPlayer}
                isCurrentTurn={isHumanTurn}
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
                drawnCards={[]}
                showCards={true}
              />
              <div className="player-label">
                <h3>Your Hand</h3>
                <span className="hand-value">Value: {CardUtils.getHandValue(humanPlayer.hand)}</span>
              </div>
            </div>
          )}

        </div>
      </div>

      {isHumanTurn && (
        <GameControls
          currentPlayer={humanPlayer}
          canDiscard={selectedCards.length > 0 && game.canDiscardCards(selectedCards)}
          canCallYaniv={game.canCallYaniv(humanPlayer.id)}
          onDiscard={handleDiscard}
          onCallYaniv={handleYaniv}
          turnPhase={gameState.turnPhase}
        />
      )}

      {gameState.gamePhase === 'roundEnd' && (
        <div className="round-end-modal">
          <div className="modal-content">
            <h2>Round Over!</h2>
            <div className="round-results">
              {gameState.players.map(player => (
                <div key={player.id} className="player-result">
                  <span>{player.name}</span>
                  <span>{player.score} points</span>
                </div>
              ))}
            </div>
            <button onClick={handleNewGame} className="new-game-btn">New Game</button>
          </div>
        </div>
      )}
    </div>
  );
}