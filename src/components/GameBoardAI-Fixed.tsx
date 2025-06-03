import React, { useState, useEffect } from 'react';
import { YanivGame } from '../game/Game';
import { ComputerAI } from '../game/ComputerAI';
import { Card, GameState } from '../types/game';
import { PlayerHand } from './PlayerHand';
import { DiscardPile } from './DiscardPile';
import { ScoreBoard } from './ScoreBoard';
import { GameControls } from './GameControls';
import { GameLog, LogEntry } from './GameLog';
import { CardUtils } from '../game/Card';

export default function GameBoardAIFixed() {
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [isAIThinking, setIsAIThinking] = useState(false);
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
    addLogEntry({
      player: 'System',
      action: 'New game started'
    });
  }, []);

  // Handle AI turn
  useEffect(() => {
    if (!game || !gameState || isAIThinking) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'AI Opponent' || gameState.gamePhase !== 'playing') return;
    
    const makeAIMove = async () => {
      setIsAIThinking(true);
      await new Promise(resolve => setTimeout(resolve, 1000)); // Delay for UX
      
      try {
        if (gameState.turnPhase === 'discard') {
          const canCallYaniv = game.canCallYaniv(currentPlayer.id);
          const decision = ComputerAI.makeDecision(
            currentPlayer.hand,
            gameState.discardPile,
            canCallYaniv
          );
          
          if (decision.action === 'yaniv') {
            const handValue = CardUtils.getHandValue(currentPlayer.hand);
            addLogEntry({
              player: 'AI Opponent',
              action: 'Called Yaniv',
              cards: currentPlayer.hand,
              details: `Hand value: ${handValue}`
            });
            game.callYaniv(currentPlayer.id);
          } else {
            // AI needs to discard
            const requiredValue = Math.floor(Math.random() * 10) + 1;
            const cardsToDiscard = ComputerAI.selectDiscard(currentPlayer.hand, requiredValue);
            
            if (cardsToDiscard) {
              addLogEntry({
                player: 'AI Opponent',
                action: 'Discarded',
                cards: cardsToDiscard,
                details: `Value: ${CardUtils.getHandValue(cardsToDiscard)}`
              });
              game.discard(currentPlayer.id, cardsToDiscard);
            } else {
              // Fallback: discard first card
              const cardToDiscard = [currentPlayer.hand[0]];
              addLogEntry({
                player: 'AI Opponent',
                action: 'Discarded',
                cards: cardToDiscard,
                details: `Value: ${cardToDiscard[0].value}`
              });
              game.discard(currentPlayer.id, cardToDiscard);
            }
          }
        } else if (gameState.turnPhase === 'draw') {
          // Decide whether to draw from deck or discard
          if (Math.random() > 0.5 && gameState.discardPile.length > 0) {
            addLogEntry({
              player: 'AI Opponent',
              action: 'Drew from discard pile'
            });
            game.drawFromDiscard(currentPlayer.id);
          } else {
            addLogEntry({
              player: 'AI Opponent',
              action: 'Drew from deck'
            });
            game.drawFromDeck(currentPlayer.id);
          }
        }
        
        setGameState(game.getState());
      } catch (error) {
        console.error('AI error:', error);
      }
      
      setIsAIThinking(false);
    };
    
    makeAIMove();
  }, [game, gameState, isAIThinking]);

  const handleDiscard = () => {
    if (!game || !gameState || selectedCards.length === 0) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      addLogEntry({
        player: 'You',
        action: 'Discarded',
        cards: selectedCards,
        details: `Value: ${CardUtils.getHandValue(selectedCards)}`
      });
      game.discard(currentPlayer.id, selectedCards);
      setSelectedCards([]);
      setGameState(game.getState());
    } catch (error: any) {
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
    } catch (error: any) {
      alert(error.message);
    }
  };

  const handleYaniv = () => {
    if (!game || !gameState) return;
    
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;
    
    try {
      const handValue = CardUtils.getHandValue(currentPlayer.hand);
      addLogEntry({
        player: 'You',
        action: 'Called Yaniv',
        cards: currentPlayer.hand,
        details: `Hand value: ${handValue}`
      });
      game.callYaniv(currentPlayer.id);
      setGameState(game.getState());
    } catch (error: any) {
      alert(error.message);
    }
  };

  const handleNewGame = () => {
    const newGame = new YanivGame(['You', 'AI Opponent']);
    setGame(newGame);
    setGameState(newGame.getState());
    setSelectedCards([]);
    setGameLog([]);
    addLogEntry({
      player: 'System',
      action: 'New game started'
    });
  };

  if (!gameState || !game) {
    return <div>Loading...</div>;
  }

  const humanPlayer = gameState.players.find(p => p.name === 'You');
  const aiPlayer = gameState.players.find(p => p.name === 'AI Opponent');
  const isHumanTurn = gameState.players[gameState.currentPlayerIndex].name === 'You';

  return (
    <div style={{ display: 'flex', gap: '20px', padding: '20px' }}>
      <div style={{ flex: 1 }}>
        <ScoreBoard players={gameState.players} currentPlayerIndex={gameState.currentPlayerIndex} />
        
        {/* AI Opponent's Hand */}
        {aiPlayer && (
          <div style={{ marginBottom: '20px' }}>
            <h3>AI Opponent {isAIThinking && '(thinking...)'}</h3>
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
          <div style={{ marginTop: '20px' }}>
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
          <div style={{ textAlign: 'center', marginTop: '20px' }}>
            <h2>Round Over!</h2>
            <button onClick={handleNewGame}>New Game</button>
          </div>
        )}
      </div>

      <GameLog entries={gameLog} />
    </div>
  );
}