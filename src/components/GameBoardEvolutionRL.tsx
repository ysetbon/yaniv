import React, { useState, useEffect, useRef, useCallback } from 'react';
import { YanivGame } from '../game/Game';
import { Card, GameState, Player } from '../types/game';
import { CardUtils } from '../game/Card';
import { PlayerHand } from './PlayerHand';
import { DiscardPile } from './DiscardPile';
import { ScoreBoard } from './ScoreBoard';
import { GameControls } from './GameControls';
import { GameLog, LogEntry } from './GameLog';
import { EvolutionRLAI } from '../game/evolution-rl/EvolutionRLAI';
import { EvolutionRLPipeline, PipelineProgress } from '../game/evolution-rl/EvolutionRLPipeline';
import './GameBoardAI.css';

type Mode = 'menu' | 'training' | 'playing';

export default function GameBoardEvolutionRL() {
  const [mode, setMode] = useState<Mode>('menu');
  const [game, setGame] = useState<YanivGame | null>(null);
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedCards, setSelectedCards] = useState<Card[]>([]);
  const [ai] = useState<EvolutionRLAI>(() => new EvolutionRLAI(0.5));
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [gameLog, setGameLog] = useState<LogEntry[]>([]);

  // Training state
  const [trainingProgress, setTrainingProgress] = useState<PipelineProgress | null>(null);
  const [trainingLog, setTrainingLog] = useState<string[]>([]);
  const pipelineRef = useRef<EvolutionRLPipeline | null>(null);

  const addLogEntry = useCallback((entry: Omit<LogEntry, 'timestamp'>) => {
    const timestamp = new Date().toLocaleTimeString();
    setGameLog((prev: LogEntry[]) => [...prev, { ...entry, timestamp }]);
  }, []);

  // Try to load pre-trained model
  useEffect(() => {
    const tryLoadModel = async () => {
      const loaded = await ai.loadModel('/models/yaniv-evolution-rl/model.json');
      if (loaded) {
        setMode('playing');
        startNewGame();
      }
    };
    tryLoadModel();
  }, []);

  const startTraining = useCallback(async () => {
    setMode('training');
    setTrainingLog([]);
    setTrainingProgress(null);

    const pipeline = new EvolutionRLPipeline({
      evolution: {
        populationSize: 30,
        generations: 50,
        gamesPerMatchup: 6,
        eliteCount: 5,
        matchupsPerPolicy: 8,
      },
      distillation: {
        gamesPerElite: 500,
        epochs: 30,
        batchSize: 64,
        learningRate: 0.001,
      },
      ppo: {
        selfPlayGamesPerBatch: 50,
        totalBatches: 40,
        ppoEpochs: 4,
        entropyCoeffStart: 0.05,
        entropyCoeffEnd: 0.01,
      },
    });
    pipelineRef.current = pipeline;

    try {
      const model = await pipeline.run((progress) => {
        setTrainingProgress(progress);
        setTrainingLog((prev: string[]) => [...prev, progress.message]);
      });

      // Set the model on the AI
      ai.setModel(model);

      // Save model for future use
      try {
        await pipeline.saveModel('indexeddb://yaniv-evolution-rl');
      } catch (e) {
        console.warn('Could not save model to indexeddb:', e);
      }

      setMode('playing');
      startNewGame();
    } catch (err) {
      console.error('Training failed:', err);
      setTrainingLog((prev: string[]) => [...prev, `ERROR: Training failed - ${err}`]);
    }
  }, [ai]);

  const startNewGame = useCallback(() => {
    const newGame = new YanivGame(['You', 'Evolution-RL AI']);
    setGame(newGame);
    setGameState(newGame.getState());
    setSelectedCards([]);
    setGameLog([]);
  }, []);

  // Handle AI turn
  useEffect(() => {
    const handleAITurn = async () => {
      if (!game || !gameState || !ai.isReady()) return;

      const currentPlayer = gameState.players[gameState.currentPlayerIndex];
      if (currentPlayer.name !== 'Evolution-RL AI' || gameState.gamePhase !== 'playing') return;

      setIsAIThinking(true);
      await new Promise(resolve => setTimeout(resolve, 800));

      try {
        const canCallYaniv = game.canCallYaniv(currentPlayer.id);

        if (gameState.turnPhase === 'discard') {
          const decision = await ai.makeDecision(
            currentPlayer.hand,
            gameState.discardPile,
            canCallYaniv,
            gameState,
            currentPlayer.id,
          );

          if (decision.action === 'yaniv') {
            const handValue = CardUtils.getHandValue(currentPlayer.hand);
            addLogEntry({
              player: 'Evolution-RL AI',
              action: 'Called Yaniv',
              cards: currentPlayer.hand,
              details: `Hand value: ${handValue}`,
            });
            game.callYaniv(currentPlayer.id);
          } else if (decision.cardsToDiscard) {
            addLogEntry({
              player: 'Evolution-RL AI',
              action: 'Discarded',
              cards: decision.cardsToDiscard,
              details: `Value: ${decision.cardsToDiscard.reduce((s: number, c: Card) => s + c.value, 0)}`,
            });
            game.discard(currentPlayer.id, decision.cardsToDiscard);
          }
        } else if (gameState.turnPhase === 'draw') {
          const decision = await ai.makeDecision(
            currentPlayer.hand,
            gameState.discardPile,
            false,
            gameState,
            currentPlayer.id,
          );

          if (decision.drawSource === 'discard') {
            addLogEntry({ player: 'Evolution-RL AI', action: 'Drew from discard pile' });
            game.drawFromDiscard(currentPlayer.id);
          } else {
            addLogEntry({ player: 'Evolution-RL AI', action: 'Drew from deck' });
            game.drawFromDeck(currentPlayer.id);
          }
        }

        setGameState(game.getState());
      } catch (error) {
        console.error('AI turn error:', error);
        // Fallback: draw from deck
        try {
          if (gameState.turnPhase === 'draw') {
            game.drawFromDeck(currentPlayer.id);
          } else {
            game.discard(currentPlayer.id, [currentPlayer.hand[0]]);
          }
          setGameState(game.getState());
        } catch (e) {
          console.error('Fallback also failed:', e);
        }
      }

      setIsAIThinking(false);
    };

    handleAITurn();
  }, [game, gameState, ai, addLogEntry]);

  const handleDiscard = () => {
    if (!game || !gameState || selectedCards.length === 0) return;
    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    if (currentPlayer.name !== 'You') return;

    try {
      addLogEntry({
        player: 'You',
        action: 'Discarded',
        cards: selectedCards,
        details: `Value: ${selectedCards.reduce((s: number, c: Card) => s + c.value, 0)}`,
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
      addLogEntry({
        player: 'You',
        action: source === 'deck' ? 'Drew from deck' : 'Drew from discard pile',
      });
      if (source === 'deck') {
        game.drawFromDeck(currentPlayer.id);
      } else {
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
      const handValue = CardUtils.getHandValue(currentPlayer.hand);
      addLogEntry({
        player: 'You',
        action: 'Called Yaniv',
        cards: currentPlayer.hand,
        details: `Hand value: ${handValue}`,
      });
      game.callYaniv(currentPlayer.id);
      setGameState(game.getState());
    } catch (error) {
      alert(error instanceof Error ? error.message : 'An error occurred');
    }
  };

  // ─── RENDER: MENU ───
  if (mode === 'menu') {
    return (
      <div className="game-board">
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '80vh', gap: '30px' }}>
          <h1 style={{ fontSize: '2.5rem', background: 'linear-gradient(45deg, #fff, #60a5fa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Yaniv - Evolution RL AI
          </h1>
          <p style={{ color: '#94a3b8', maxWidth: '600px', textAlign: 'center', lineHeight: '1.6' }}>
            Three-phase AI training: Genetic Evolution generates elite heuristic policies,
            Distillation trains a neural network to imitate them, then PPO self-play
            pushes beyond the genetic ceiling.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '300px' }}>
            <button
              onClick={startTraining}
              style={{ padding: '15px 30px', fontSize: '1.2rem', fontWeight: 'bold', border: 'none', borderRadius: '12px', background: 'linear-gradient(135deg, #7c3aed, #2563eb)', color: 'white', cursor: 'pointer' }}
            >
              Train New AI
            </button>
            <div style={{ textAlign: 'center', color: '#64748b', fontSize: '0.85rem' }}>
              Phase 1: Evolve 30 policies x 50 generations<br/>
              Phase 2: Distill into neural network<br/>
              Phase 3: PPO self-play refinement
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ─── RENDER: TRAINING ───
  if (mode === 'training') {
    const phaseColors: Record<string, string> = {
      evolution: '#7c3aed',
      distillation: '#2563eb',
      ppo: '#059669',
      done: '#10b981',
      idle: '#64748b',
    };
    const phaseNames: Record<string, string> = {
      evolution: 'Phase 1: Genetic Evolution',
      distillation: 'Phase 2: Distillation',
      ppo: 'Phase 3: PPO Fine-tuning',
      done: 'Complete',
      idle: 'Initializing...',
    };
    const currentPhase = trainingProgress?.phase || 'idle';

    return (
      <div className="game-board">
        <div style={{ maxWidth: '800px', margin: '0 auto', padding: '40px 20px' }}>
          <h1 style={{ fontSize: '2rem', marginBottom: '30px', background: 'linear-gradient(45deg, #fff, #60a5fa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            Training Evolution-RL AI
          </h1>

          {/* Phase indicator */}
          <div style={{ display: 'flex', gap: '10px', marginBottom: '30px' }}>
            {['evolution', 'distillation', 'ppo'].map((phase, i) => {
              const isActive = currentPhase === phase;
              const isDone = ['evolution', 'distillation', 'ppo'].indexOf(currentPhase) > i || currentPhase === 'done';
              return (
                <div
                  key={phase}
                  style={{
                    flex: 1,
                    padding: '12px',
                    borderRadius: '10px',
                    background: isActive
                      ? `linear-gradient(135deg, ${phaseColors[phase]}88, ${phaseColors[phase]}44)`
                      : isDone
                        ? 'rgba(16, 185, 129, 0.2)'
                        : 'rgba(255, 255, 255, 0.05)',
                    border: isActive ? `2px solid ${phaseColors[phase]}` : '2px solid transparent',
                    textAlign: 'center',
                    color: isActive ? '#fff' : isDone ? '#10b981' : '#64748b',
                    fontSize: '0.85rem',
                    fontWeight: isActive ? 600 : 400,
                  }}
                >
                  {isDone && !isActive ? 'Done' : phaseNames[phase]}
                </div>
              );
            })}
          </div>

          {/* Progress bar */}
          {trainingProgress && (
            <div style={{ marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: '#e0e7ff', fontWeight: 600 }}>{phaseNames[currentPhase]}</span>
                <span style={{ color: '#94a3b8' }}>{(trainingProgress.phaseProgress * 100).toFixed(0)}%</span>
              </div>
              <div style={{ height: '8px', borderRadius: '4px', background: 'rgba(255,255,255,0.1)', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${trainingProgress.phaseProgress * 100}%`,
                  background: `linear-gradient(90deg, ${phaseColors[currentPhase]}, ${phaseColors[currentPhase]}cc)`,
                  borderRadius: '4px',
                  transition: 'width 0.3s',
                }} />
              </div>
            </div>
          )}

          {/* Training log */}
          <div style={{
            background: 'rgba(0, 0, 0, 0.3)',
            borderRadius: '12px',
            padding: '20px',
            maxHeight: '400px',
            overflowY: 'auto',
            fontFamily: 'monospace',
            fontSize: '0.85rem',
            lineHeight: '1.8',
          }}>
            {trainingLog.length === 0 && (
              <div style={{ color: '#64748b' }}>Initializing training pipeline...</div>
            )}
            {trainingLog.map((line: string, i: number) => (
              <div key={i} style={{ color: line.includes('ERROR') ? '#ef4444' : line.includes('complete') ? '#10b981' : '#cbd5e1' }}>
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ─── RENDER: PLAYING ───
  if (!gameState || !game) {
    return (
      <div style={{ padding: '20px', background: 'white', color: 'black', minHeight: '200px' }}>
        <h2>Loading game...</h2>
      </div>
    );
  }

  const humanPlayer = gameState.players.find((p: Player) => p.name === 'You');
  const aiPlayer = gameState.players.find((p: Player) => p.name === 'Evolution-RL AI');
  const isHumanTurn = gameState.players[gameState.currentPlayerIndex].name === 'You';

  return (
    <div className="game-board">
      <div className="game-header">
        <h1>Yaniv - Evolution RL AI</h1>
        <div className="ai-mode-selector">
          <span className="model-status">Evolution-RL Ready</span>
        </div>
      </div>

      <div className="game-layout">
        <div className="left-panel">
          <ScoreBoard players={gameState.players} currentPlayerId={gameState.players[gameState.currentPlayerIndex].id} />
          <GameLog entries={gameLog} />
        </div>

        <div className="game-table">
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

          {humanPlayer && (
            <div className="human-player-area">
              <PlayerHand
                player={humanPlayer}
                isCurrentTurn={isHumanTurn}
                onCardSelect={(card) => {
                  if (!isHumanTurn) return;
                  const isSelected = selectedCards.some((c: Card) => c.suit === card.suit && c.rank === card.rank);
                  if (isSelected) {
                    setSelectedCards(selectedCards.filter((c: Card) => !(c.suit === card.suit && c.rank === card.rank)));
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
          canCallYaniv={game.canCallYaniv(humanPlayer!.id)}
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
              {gameState.players.map((player: Player) => (
                <div key={player.id} className="player-result">
                  <span>{player.name}</span>
                  <span>{player.score} points</span>
                </div>
              ))}
            </div>
            <button onClick={startNewGame} className="new-game-btn">New Game</button>
          </div>
        </div>
      )}
    </div>
  );
}
