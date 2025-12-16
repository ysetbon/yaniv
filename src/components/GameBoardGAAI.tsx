import React, { useState, useEffect } from 'react';
import { YanivGame } from '../game/Game';
import { GATrainedAIWrapper } from '../game/GATrainedAIWrapper';
import { Card, GameState } from '../types/game';
import { CardUtils } from '../game/Card';
import { PlayerHand } from './PlayerHand';
import { DiscardPile } from './DiscardPile';
import { ScoreBoard } from './ScoreBoard';
import { GameControls } from './GameControls';
import { GameLog, LogEntry } from './GameLog';
import './GameBoardAI.css';

export default function GameBoardGAAI() {
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [ai, setAI] = useState<GATrainedAIWrapper | null>(null);
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [gameLog, setGameLog] = useState<LogEntry[]>([]);

  const addLogEntry = (entry: Omit<LogEntry, 'timestamp'>) => {
    const timestamp = new Date().toLocaleTimeString();
    setGameLog(prev => [...prev, { ...entry, timestamp }]);
  };

  // Initialize game
  useEffect(() => {
    const newGame = new YanivGame(['You', 'GA AI']);
    setGame(newGame);
    setGameState(newGame.getState());
  }, []);

  // Initialize AI
  useEffect(() => {
    const initAI = async () => {
      console.log('Initializing GA-trained AI...');
      const newAI = new GATrainedAIWrapper();

      try {
        await newAI.loadModel('/src/game/yaniv_model.json');
        setModelLoaded(true);
        setAI(newAI);
        console.log('GA AI model loaded successfully');
        addLogEntry({
          player: 'System',
          action: 'AI Loaded',
          details: 'GA-trained neural network loaded'
        });
      } catch (error) {
        console.error('Error loading GA AI model:', error);
        setModelLoaded(false);
        addLogEntry({
          player: 'System',
          action: 'AI Error',
          details: 'Failed to load GA AI model'
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
      if (currentPlayer.name !== 'GA AI' || gameState.gamePhase !== 'playing') return;

      setIsAIThinking(true);

      // Add delay for better UX
      await new Promise(resolve => setTimeout(resolve, 800));

      try {
        const aiPlayerIndex = gameState.players.findIndex(p => p.name === 'GA AI');
        const decision = ai.makeMove(gameState, aiPlayerIndex);

        if (decision.action === 'yaniv') {
          const handValue = CardUtils.getHandValue(currentPlayer.hand);
          addLogEntry({
            player: 'GA AI',
            action: 'Called Yaniv',
            cards: currentPlayer.hand,
            details: `Hand value: ${handValue}`
          });
          game.callYaniv(currentPlayer.id);
        } else if (decision.action === 'draw') {
          const drawnCards = game.drawCards(currentPlayer.id, decision.source || 'deck');
          addLogEntry({
            player: 'GA AI',
            action: `Drew from ${decision.source || 'deck'}`,
            cards: drawnCards
          });
        } else if (decision.action === 'discard' && decision.cards) {
          addLogEntry({
            player: 'GA AI',
            action: 'Discarded',
            cards: decision.cards
          });
          game.discardCards(currentPlayer.id, decision.cards);
        }

        setGameState(game.getState());
      } catch (error) {
        console.error('AI decision error:', error);
      }

      setIsAIThinking(false);
    };

    const timeoutId = setTimeout(handleAITurn, 100);
    return () => clearTimeout(timeoutId);
  }, [game, gameState, ai, modelLoaded]);

  const handleCardSelect = (card: Card) => {
    if (!gameState || gameState.turnPhase !== 'discard') return;

    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;

    setSelectedCards(prev => {
      const isSelected = prev.some(c => CardUtils.areEqual(c, card));
      if (isSelected) {
        return prev.filter(c => !CardUtils.areEqual(c, card));
      }
      return [...prev, card];
    });
  };

  const handleDiscard = () => {
    if (!game || !gameState || selectedCards.length === 0) return;

    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;

    if (!CardUtils.isValidDiscard(selectedCards)) {
      alert('Invalid discard! Must be single card, set (same rank), or run (consecutive same suit)');
      return;
    }

    addLogEntry({
      player: 'You',
      action: 'Discarded',
      cards: selectedCards
    });

    game.discardCards(currentPlayer.id, selectedCards);
    setSelectedCards([]);
    setGameState(game.getState());
  };

  const handleDraw = (source: 'deck' | 'discard') => {
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
  };

  const handleYaniv = () => {
    if (!game || !gameState) return;

    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;

    const handValue = CardUtils.getHandValue(currentPlayer.hand);
    addLogEntry({
      player: 'You',
      action: 'Called Yaniv',
      cards: currentPlayer.hand,
      details: `Hand value: ${handValue}`
    });

    game.callYaniv(currentPlayer.id);
    setGameState(game.getState());
  };

  const handleNewGame = () => {
    const newGame = new YanivGame(['You', 'GA AI']);
    setGame(newGame);
    setGameState(newGame.getState());
    setSelectedCards([]);
    setGameLog([]);
    addLogEntry({
      player: 'System',
      action: 'New Game',
      details: 'Game started'
    });
  };

  if (!gameState) {
    return <div className="loading">Loading game...</div>;
  }

  const humanPlayer = gameState.players.find(p => p.name === 'You')!;
  const aiPlayer = gameState.players.find(p => p.name === 'GA AI')!;
  const isHumanTurn = gameState.players[gameState.currentPlayerIndex].name === 'You';
  const canCallYaniv = isHumanTurn &&
    gameState.turnPhase === 'discard' &&
    CardUtils.getHandValue(humanPlayer.hand) <= 7 &&
    !humanPlayer.hasCalledYanivLastRound;

  return (
    <div className="game-board-ai">
      <div className="game-header">
        <h1>Yaniv vs GA-Trained AI</h1>
        {!modelLoaded && <p className="warning">AI model not loaded - using fallback</p>}
        {isAIThinking && <p className="thinking">AI is thinking...</p>}
      </div>

      <ScoreBoard players={gameState.players} />

      <div className="game-area">
        <div className="opponent-area">
          <h3>GA AI ({aiPlayer.hand.length} cards)</h3>
          <div className="card-backs">
            {aiPlayer.hand.map((_, i) => (
              <div key={i} className="card-back">🂠</div>
            ))}
          </div>
        </div>

        <div className="center-area">
          <DiscardPile cards={gameState.discardPile} />

          <div className="game-info">
            <p>Round: {gameState.roundNumber}</p>
            <p>Phase: {gameState.turnPhase}</p>
            <p>Turn: {gameState.players[gameState.currentPlayerIndex].name}</p>
          </div>
        </div>

        <div className="player-area">
          <h3>Your Hand (Value: {CardUtils.getHandValue(humanPlayer.hand)})</h3>
          <PlayerHand
            cards={humanPlayer.hand}
            selectedCards={selectedCards}
            onCardSelect={handleCardSelect}
            disabled={!isHumanTurn || gameState.turnPhase !== 'discard'}
          />
        </div>
      </div>

      <GameControls
        canDiscard={isHumanTurn && gameState.turnPhase === 'discard' && selectedCards.length > 0}
        canDraw={isHumanTurn && gameState.turnPhase === 'draw'}
        canYaniv={canCallYaniv}
        onDiscard={handleDiscard}
        onDrawDeck={() => handleDraw('deck')}
        onDrawDiscard={() => handleDraw('discard')}
        onYaniv={handleYaniv}
        onNewGame={handleNewGame}
        gameOver={gameState.gamePhase === 'gameEnd'}
        winner={gameState.winner}
        players={gameState.players}
      />

      <GameLog entries={gameLog} />

      {gameState.gamePhase === 'gameEnd' && (
        <div className="game-over-overlay">
          <div className="game-over-modal">
            <h2>Game Over!</h2>
            <p>Winner: {gameState.players.find(p => p.id === gameState.winner)?.name}</p>
            <button onClick={handleNewGame}>Play Again</button>
          </div>
        </div>
      )}
    </div>
  );
}
