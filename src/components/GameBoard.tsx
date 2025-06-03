import { useState, useEffect, useCallback } from 'react';
import { YanivGame } from '../game/Game';
import { Card, GameState, Player } from '../types/game';
import { CardComponent } from './Card';
import { PlayerHand } from './PlayerHand';
import { DiscardPile } from './DiscardPile';
import { ScoreBoard } from './ScoreBoard';
import { GameControls } from './GameControls';
import { CardUtils } from '../game/Card';
import { ComputerAI } from '../game/ComputerAI';
import { EnhancedComputerAI } from '../game/EnhancedComputerAI';
import { GameLog, LogEntry } from './GameLog';
import { MoveAnalyzer } from '../game/MoveAnalyzer';
import { DevLogger } from './DevLogger';

interface GameBoardProps {
  playerNames: string[];
  onEndGame: () => void;
}

export function GameBoard({ playerNames, onEndGame }: GameBoardProps) {
  // Initialize the enhanced AI with neural network mode
  const [computerAI] = useState(() => new EnhancedComputerAI('neural-network'));
  const [game] = useState(() => new YanivGame(playerNames));
  const [gameState, setGameState] = useState<GameState>(game.getState());
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [message, setMessage] = useState<string>('');
  const [messageTimeout, setMessageTimeout] = useState<NodeJS.Timeout | null>(null);
  const [gameLog, setGameLog] = useState<LogEntry[]>([]);

  const humanPlayer = gameState.players[0];
  const computerPlayer = gameState.players[1];
  const currentPlayer = gameState.players[gameState.currentPlayerIndex];
  const isHumanTurn = currentPlayer.id === humanPlayer.id;

  // Helper function to add log entry
  const addLogEntry = useCallback((entry: Omit<LogEntry, 'timestamp'>) => {
    const timestamp = new Date().toLocaleTimeString();
    setGameLog(prev => [...prev, { ...entry, timestamp }]);
  }, []);

  // Auto-hide messages after 3 seconds
  const showMessage = useCallback((msg: string, duration: number = 3000) => {
    setMessage(msg);
    if (messageTimeout) {
      clearTimeout(messageTimeout);
    }
    const timeout = setTimeout(() => {
      setMessage('');
    }, duration);
    setMessageTimeout(timeout);
  }, [messageTimeout]);

  const executeComputerTurn = useCallback(async () => {
    if (!isHumanTurn && gameState.gamePhase === 'playing') {
      // Prevent multiple executions
      if ((window as any).__computerTurnInProgress) {
        return;
      }
      (window as any).__computerTurnInProgress = true;
      
      showMessage('Computer is thinking...', 1500);
      
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Get fresh state and computer player
      const currentState = game.getState();
      const currentComputerPlayer = currentState.players[1];

      if (currentState.turnPhase === 'discard') {
        // Log computer's current hand
        console.log('\n=== COMPUTER TURN ===');
        console.log('Computer hand:', currentComputerPlayer.hand.map(c => `${c.rank}${c.suit}`).join(', '));
        console.log('Hand value:', CardUtils.getHandValue(currentComputerPlayer.hand));
        console.log('Can call Yaniv:', game.canCallYaniv(currentComputerPlayer.id));
        
        // Computer decides what to discard
        const humanHandSize = currentState.players.find(p => p.id === humanPlayer.id)?.hand.length || 0;
        const decision = await computerAI.makeDecision(
          currentComputerPlayer.hand,
          currentState.discardPile,
          game.canCallYaniv(currentComputerPlayer.id),
          currentState,
          currentComputerPlayer.id
        );

        if (decision.action === 'yaniv') {
          const handValue = CardUtils.getHandValue(currentComputerPlayer.hand);
          addLogEntry({
            player: 'Computer',
            action: 'Called Yaniv',
            cards: currentComputerPlayer.hand,
            details: `Hand value: ${handValue}`
          });
          const result = game.callYaniv(currentComputerPlayer.id);
          if (result.success) {
            if (result.assaf) {
              showMessage('Assaf! Computer gets 30 penalty points!', 5000);
            } else {
              showMessage('Computer successfully called Yaniv!', 5000);
            }
            setGameState(game.getState());
            handleRoundEnd();
            return;
          }
        }

        // Find a valid discard - improved selection
        let actualDiscard: Card[] | null = null;
        const validDiscards = findValidDiscards(currentComputerPlayer.hand);
        console.log('Valid discards found:', validDiscards.length);
        
        if (validDiscards.length > 0) {
          // Sort discards by strategic value
          const scoredDiscards = validDiscards.map(discard => {
            const discardValue = CardUtils.getHandValue(discard);
            const remainingHand = currentComputerPlayer.hand.filter(c => 
              !discard.some(d => CardUtils.areEqual(c, d))
            );
            const remainingValue = CardUtils.getHandValue(remainingHand);
            
            // Score: prefer high discard value and low remaining value
            let score = discardValue * 0.5 - remainingValue * 0.5;
            
            // Bonus for sets/runs (they count as 0 in hand)
            if (discard.length > 1 && (CardUtils.isSet(discard) || CardUtils.isRun(discard))) {
              score += 10;
            }
            
            // Penalty for discarding cards that could form sets
            for (const card of discard) {
              const matchingCards = remainingHand.filter(c => c.rank === card.rank);
              if (matchingCards.length >= 1) {
                // We're breaking up a potential set
                score -= 5;
              }
            }
            
            // Penalty for discarding cards that could form runs
            for (const card of discard) {
              const sameSuitCards = remainingHand.filter(c => c.suit === card.suit);
              for (const suitCard of sameSuitCards) {
                const diff = Math.abs(suitCard.value - card.value);
                if (diff <= 2) {
                  // Close to forming a run
                  score -= 3;
                }
              }
            }
            
            return { discard, score };
          });
          
          // Sort by score (highest first)
          scoredDiscards.sort((a, b) => b.score - a.score);
          actualDiscard = scoredDiscards[0].discard;
        }
        
        // Execute the discard and log only the actual move
        if (actualDiscard) {
          const handBefore = [...currentComputerPlayer.hand];
          console.log('Attempting to discard:', actualDiscard.map(c => `${c.rank}${c.suit}`).join(', '));
          
          try {
            const success = game.discard(currentComputerPlayer.id, actualDiscard);
            console.log('Discard success:', success);
            
            const remainingHand = handBefore.filter(c => !actualDiscard!.some(d => CardUtils.areEqual(c, d)));
            console.log('Computer discards:', actualDiscard.map(c => `${c.rank}${c.suit}`).join(', '));
            console.log('Discard value:', CardUtils.getHandValue(actualDiscard));
            console.log('Remaining hand value:', CardUtils.getHandValue(remainingHand));
            
            // Analyze the discard
            const analysis = MoveAnalyzer.analyzeDiscard(
              handBefore,
              actualDiscard,
              remainingHand
            );
            console.log('Move analysis:', analysis);
            
            addLogEntry({
              player: 'Computer',
              action: 'Discarded',
              cards: actualDiscard,
              details: `Value: ${CardUtils.getHandValue(actualDiscard)}`
            });
          } catch (error) {
            console.log('Discard failed:', error);
          }
        } else {
          console.log('No valid discard found!');
        }
        
        const newState = game.getState();
        setGameState(newState);
        showMessage(`Computer discarded cards`, 2000);
        
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      if (game.getState().turnPhase === 'draw') {
        // Computer draws a card
        // Check if computer can draw from discard (not their own card)
        const opponentDiscard = game.getLastOpponentDiscard(currentComputerPlayer.id);
        const canDrawDiscard = opponentDiscard !== null;
        
        let visibleCard: Card | null = null;
        if (canDrawDiscard && opponentDiscard && opponentDiscard.length > 0) {
          visibleCard = opponentDiscard[0];
          console.log('\nVisible discard card:', `${visibleCard.rank}${visibleCard.suit}`);
          console.log('Card value:', visibleCard.value);
        }
        
        // Use strategic draw decision for draw phase
        // For now, use the rule-based strategic draw since neural network handles discard/yaniv
        const drawDecision = ComputerAI.makeStrategicDrawDecision(
          currentComputerPlayer.hand,
          visibleCard
        );
        
        console.log('Draw decision:', drawDecision.drawSource, '-', drawDecision.reasoning);
        
        if (drawDecision.drawSource === 'deck' || !canDrawDiscard) {
          console.log('Computer draws from DECK (hidden card)');
          addLogEntry({
            player: 'Computer',
            action: 'Drew from deck',
            details: drawDecision.reasoning
          });
          const drawnCards = game.drawFromDeck(currentComputerPlayer.id);
          console.log('Drew:', drawnCards.map(c => `${c.rank}${c.suit}`).join(', '));
        } else {
          try {
            console.log('Computer draws from DISCARD PILE');
            addLogEntry({
              player: 'Computer',
              action: 'Drew from discard pile',
              details: drawDecision.reasoning
            });
            const drawnCards = game.drawFromDiscard(currentComputerPlayer.id);
            console.log('Drew:', drawnCards.map(c => `${c.rank}${c.suit}`).join(', '));
          } catch (e) {
            // If can't draw from discard, draw from deck
            console.log('Computer draws from DECK (discard failed)');
            addLogEntry({
              player: 'Computer',
              action: 'Drew from deck',
              details: 'Could not draw from discard'
            });
            const drawnCards = game.drawFromDeck(currentComputerPlayer.id);
            console.log('Drew:', drawnCards.map(c => `${c.rank}${c.suit}`).join(', '));
          }
        }
        
        // Log new hand after drawing
        const newComputerPlayer = game.getState().players[1];
        console.log('New hand:', newComputerPlayer.hand.map(c => `${c.rank}${c.suit}`).join(', '));
        console.log('New hand value:', CardUtils.getHandValue(newComputerPlayer.hand));
        
        // Analyze the draw decision
        const drawAnalysis = MoveAnalyzer.analyzeDrawDecision(
          visibleCard,
          game.getState().turnPhase !== 'draw', // If turn phase changed, it means draw was successful
          currentComputerPlayer.hand
        );
        console.log('Draw analysis:', drawAnalysis);
        console.log('===================\n');

        setGameState(game.getState());
        showMessage('');
      }
      
      // Clear the flag
      (window as any).__computerTurnInProgress = false;
    }
  }, [isHumanTurn, gameState, computerPlayer, game, showMessage]);

  useEffect(() => {
    executeComputerTurn();
  }, [executeComputerTurn]);

  const handleRoundEnd = () => {
    setTimeout(() => {
      if (game.getState().gamePhase === 'gameEnd') {
        const winner = game.getState().players.find(p => p.id === game.getState().winner);
        showMessage(`Game Over! ${winner?.name} wins!`, 10000);
      } else {
        showMessage('Starting new round...', 2000);
        setSelectedCards([]);
        setGameState(game.getState());
      }
    }, 3000);
  };

  const handleCardSelect = (card: Card) => {
    if (!isHumanTurn || gameState.turnPhase !== 'discard') return;

    const isSelected = selectedCards.some(c => CardUtils.areEqual(c, card));
    if (isSelected) {
      setSelectedCards(selectedCards.filter(c => !CardUtils.areEqual(c, card)));
    } else {
      setSelectedCards([...selectedCards, card]);
    }
  };

  const handleDiscard = () => {
    if (selectedCards.length === 0 || gameState.turnPhase !== 'discard') return;

    try {
      if (game.discard(humanPlayer.id, selectedCards)) {
        addLogEntry({
          player: playerNames[0],
          action: 'Discarded',
          cards: selectedCards,
          details: `Value: ${CardUtils.getHandValue(selectedCards)}`
        });
        setGameState(game.getState());
        setSelectedCards([]);
        showMessage('Cards discarded. Now draw one card.');
      } else {
        showMessage('Invalid discard - must be valid single card, set, or run', 2000);
      }
    } catch (error) {
      showMessage('Cannot discard those cards', 2000);
    }
  };

  const handleDrawFromDeck = () => {
    if (gameState.turnPhase !== 'draw' || !isHumanTurn) return;

    try {
      addLogEntry({
        player: playerNames[0],
        action: 'Drew from deck'
      });
      game.drawFromDeck(humanPlayer.id);
      setGameState(game.getState());
      showMessage('Drew 1 card from deck');
    } catch (error) {
      showMessage('Must discard first', 2000);
    }
  };

  const handleDrawFromDiscard = () => {
    if (gameState.turnPhase !== 'draw' || !isHumanTurn) return;

    try {
      addLogEntry({
        player: playerNames[0],
        action: 'Drew from discard pile'
      });
      game.drawFromDiscard(humanPlayer.id);
      setGameState(game.getState());
      showMessage('Drew card from discard pile');
    } catch (error: any) {
      showMessage(error.message || 'Cannot draw from discard', 2000);
    }
  };

  const handleCallYaniv = () => {
    const handValue = CardUtils.getHandValue(humanPlayer.hand);
    addLogEntry({
      player: playerNames[0],
      action: 'Called Yaniv',
      cards: humanPlayer.hand,
      details: `Hand value: ${handValue}`
    });
    const result = game.callYaniv(humanPlayer.id);
    if (result.success) {
      if (result.assaf) {
        showMessage(`Assaf! You get 30 penalty points!`, 5000);
      } else {
        showMessage(`You successfully called Yaniv!`, 5000);
      }
      setGameState(game.getState());
      handleRoundEnd();
    }
  };

  const findValidDiscards = (hand: Card[]): Card[][] => {
    const validDiscards: Card[][] = [];
    
    // Single cards
    hand.forEach(card => validDiscards.push([card]));
    
    // Sets
    const byRank = new Map<string, Card[]>();
    hand.forEach(card => {
      const cards = byRank.get(card.rank) || [];
      cards.push(card);
      byRank.set(card.rank, cards);
    });
    
    byRank.forEach(cards => {
      if (cards.length >= 2) {
        for (let i = 2; i <= cards.length; i++) {
          validDiscards.push(cards.slice(0, i));
        }
      }
    });
    
    // Runs
    const bySuit = new Map<string, Card[]>();
    hand.forEach(card => {
      const cards = bySuit.get(card.suit) || [];
      cards.push(card);
      bySuit.set(card.suit, cards);
    });
    
    bySuit.forEach(cards => {
      const sorted = cards.sort((a, b) => a.value - b.value);
      for (let start = 0; start < sorted.length - 2; start++) {
        for (let end = start + 2; end < sorted.length; end++) {
          const run = sorted.slice(start, end + 1);
          if (CardUtils.isRun(run)) {
            validDiscards.push(run);
          }
        }
      }
    });
    
    return validDiscards;
  };

  // Always get fresh opponent discard from game state
  const opponentDiscards = game.getLastOpponentDiscard(humanPlayer.id);
  const opponentCard = opponentDiscards && opponentDiscards.length > 0 ? opponentDiscards[0] : null;
  const canDrawFromDiscard = isHumanTurn && gameState.turnPhase === 'draw' && opponentCard !== null;

  return (
    <div className="game-board">
      <GameLog entries={gameLog} />
      <DevLogger />
      
      <div className="game-header">
        <h1>Yaniv</h1>
        <button className="quit-button" onClick={onEndGame}>Quit Game</button>
      </div>
      
      <ScoreBoard players={gameState.players} currentPlayerId={currentPlayer.id} />
      
      <div className="game-area">
        <div className="computer-area">
          <div className="computer-hand">
            <h3>{computerPlayer.name}'s Hand</h3>
            <div className="hidden-cards">
              {computerPlayer.hand.map((_, index) => (
                <CardComponent key={index} card={{} as Card} faceDown={true} />
              ))}
            </div>
            <span className="card-count">{computerPlayer.hand.length} cards</span>
          </div>
        </div>

        <div className="center-area">
          <div 
            className={`deck ${gameState.turnPhase === 'draw' && isHumanTurn ? 'clickable' : ''}`} 
            onClick={handleDrawFromDeck}
          >
            <div className="deck-cards">
              <div className="card-back"></div>
              <div className="card-back"></div>
              <div className="card-back"></div>
            </div>
            <span>Draw Deck</span>
          </div>
          
          <DiscardPile 
            opponentCard={opponentCard}
            totalCards={gameState.discardPile.length}
            onDrawCard={handleDrawFromDiscard}
            canDraw={canDrawFromDiscard}
            isDrawPhase={isHumanTurn && gameState.turnPhase === 'draw'}
            isHumanTurn={isHumanTurn}
          />
        </div>
        
        <div className="player-area">
          <PlayerHand
            player={humanPlayer}
            isCurrentTurn={isHumanTurn}
            selectedCards={selectedCards}
            onCardSelect={(card) => handleCardSelect(card)}
            drawnCards={[]}
            showCards={true}
          />
        </div>
        
        {isHumanTurn && (
          <GameControls
            currentPlayer={humanPlayer}
            canDiscard={selectedCards.length > 0 && gameState.turnPhase === 'discard'}
            canCallYaniv={game.canCallYaniv(humanPlayer.id) && gameState.turnPhase === 'discard'}
            onDiscard={handleDiscard}
            onCallYaniv={handleCallYaniv}
            turnPhase={gameState.turnPhase}
          />
        )}
        
        {message && (
          <div className="message-box">
            {message}
          </div>
        )}
      </div>
    </div>
  );
}