import * as tf from '@tensorflow/tfjs';
import { YanivGame } from './Game';
import { NeuralNetworkAI } from './NeuralNetworkAI';
import { Card } from '../types/game';
import { CardUtils } from './Card';
import { RewardStrategy } from './RewardStrategy';
import { ValidMoveTeacher } from './ValidMoveTeacher';

interface Experience {
  state: tf.Tensor2D;
  action: number;
  reward: number;
  nextState: tf.Tensor2D | null;
  done: boolean;
}

export class TrainingSystemV2 {
  private experiences: Experience[] = [];
  private readonly maxExperiences = 50000;  // Increased for longer training sessions
  private readonly batchSize = 32;
  private readonly gamma = 0.95;
  private neuralAI: NeuralNetworkAI;
  
  constructor() {
    console.log('Initializing TrainingSystemV2...');
    this.neuralAI = new NeuralNetworkAI();
    console.log('Neural AI initialized');
  }
  
  async runSelfPlayEpisode(): Promise<void> {
    const players = ['AI1', 'AI2', 'AI3', 'AI4'];
    const game = new YanivGame(players);
    const episodeExperiences: Experience[] = [];
    let turnCount = 0;
    let roundEnded = false;
    
    console.log('Starting self-play episode...');
    
    // Log initial hand values
    const initialState = game.getState();
    const initialHandValues = initialState.players.map(p => `${p.id}: ${CardUtils.getHandValue(p.hand)}`).join(', ');
    console.log(`Initial hand values: ${initialHandValues}`);
    
    while (game.getState().gamePhase === 'playing' && !roundEnded) {
      const state = game.getState();
      const currentPlayer = state.players[state.currentPlayerIndex];
      const playerId = currentPlayer.id;
      
      // Encode current state first
      const stateTensor = this.neuralAI.encodeGameState(
        currentPlayer.hand,
        state.discardPile,
        state,
        playerId
      );
      
      // Safety check to prevent infinite loops
      if (turnCount > 500) {
        console.log(`Episode exceeded 500 turns, forcing end...`);
        // Force the episode to end with current hand values as final state
        roundEnded = true;
        
        // Give negative rewards for not finishing the game
        episodeExperiences.push({
          state: stateTensor,
          action: 0,
          reward: -50, // Penalty for not finishing
          nextState: null,
          done: true
        });
        break;
      }
      
      let actionIndex = 0;
      let reward = 0;
      
      try {
        turnCount++;
        
        // Log hand values periodically
        if (turnCount % 100 === 0) {
          const handValues = state.players.map(p => `${p.id}: ${CardUtils.getHandValue(p.hand)}`).join(', ');
          console.log(`Turn ${turnCount} - Hand values: ${handValues}`);
        }
        
        // For training, sometimes try invalid moves to learn they're bad
        const exploreInvalidMoves = Math.random() < 0.05; // 5% of the time
        
        // Check if can call Yaniv (only during discard phase)
        if (state.turnPhase === 'discard' && game.canCallYaniv(playerId)) {
          const handValue = CardUtils.getHandValue(currentPlayer.hand);
          
          // Use neural network to decide whether to call Yaniv
          if (turnCount % 50 === 0) {
            console.log(`Turn ${turnCount}: ${playerId} can call Yaniv with hand value ${handValue}`);
          }
          const decision = await this.neuralAI.makeDecision(
            currentPlayer.hand,
            state.discardPile,
            state,
            playerId
          );
          
          // During training, sometimes force Yaniv to explore this action
          // Increase probability as turns increase to ensure episodes end
          const yanivProbability = Math.min(0.3 + (turnCount / 1000) * 0.5, 0.8);
          const shouldForceYaniv = Math.random() < yanivProbability;
          
          if (decision.action === 'yaniv' || shouldForceYaniv) {
            console.log(`Turn ${turnCount}: ${playerId} calls Yaniv with hand value ${handValue} (forced: ${shouldForceYaniv})`);
            const result = game.callYaniv(playerId);
            actionIndex = 4;
            reward = RewardStrategy.calculateYanivReward(
              handValue, 
              result.success, 
              result.assaf || false
            );
            
            // Store experience and break as round ended
            episodeExperiences.push({
              state: stateTensor,
              action: actionIndex,
              reward: reward,
              nextState: null,
              done: true
            });
            roundEnded = true;
            break;
          }
        }
        
        // Handle turn phases
        if (state.turnPhase === 'discard') {
          // Generate training examples including invalid moves
          const trainingExamples = ValidMoveTeacher.generateTrainingExamples(currentPlayer.hand);
          
          // During training, sometimes try invalid moves to learn they're bad
          const shouldExploreInvalid = Math.random() < 0.1; // 10% of the time
          
          let selectedExample: any = null;
          
          if (shouldExploreInvalid && trainingExamples.some(e => !e.valid)) {
            // Pick a random invalid move to learn it's bad
            const invalidExamples = trainingExamples.filter(e => !e.valid);
            selectedExample = invalidExamples[Math.floor(Math.random() * invalidExamples.length)];
            console.log(`Turn ${turnCount}: ${playerId} tries INVALID discard: ${selectedExample.reason}`);
          } else {
            // Normal selection from valid moves
            const validExamples = trainingExamples.filter(e => e.valid);
            
            // Select based on score with some exploration
            for (const example of validExamples) {
              const score = example.score + Math.random() * 2; // Add exploration
              if (!selectedExample || score > selectedExample.score) {
                selectedExample = example;
              }
            }
          }
          
          // Try to execute the selected discard
          if (selectedExample) {
            const handBefore = [...currentPlayer.hand];
            
            try {
              const success = game.discard(playerId, selectedExample.cards);
              
              if (success && selectedExample.valid) {
                // Valid move succeeded
                const handAfter = game.getState().players[state.currentPlayerIndex].hand;
                actionIndex = 5; // Simplified action mapping
                reward = RewardStrategy.calculateDiscardReward(
                  handBefore,
                  selectedExample.cards,
                  handAfter
                );
                
                if (turnCount % 20 === 0) {
                  const newHandValue = CardUtils.getHandValue(handAfter);
                  const oldHandValue = CardUtils.getHandValue(handBefore);
                  console.log(`Turn ${turnCount}: ${playerId} discards ${selectedExample.cards.map(c => c.rank + c.suit).join(', ')} (${selectedExample.reason}, hand: ${oldHandValue} -> ${newHandValue}, reward: ${reward.toFixed(2)})`);
                }
              } else if (!success && !selectedExample.valid) {
                // Invalid move correctly failed - good learning!
                reward = -Infinity; // NEVER do this
                actionIndex = 5;
                console.log(`LEARNED: Invalid discard rejected - ${selectedExample.reason}`);
              } else if (success && !selectedExample.valid) {
                // This shouldn't happen - our validation is wrong
                console.error('WARNING: Invalid move succeeded!', selectedExample);
                reward = -Infinity;
              } else {
                // Valid move failed - something's wrong
                console.error('Valid move failed:', selectedExample);
                reward = -Infinity;
              }
            } catch (error) {
              // Move failed - learn from it
              reward = selectedExample.valid ? -30 : -Infinity;
              console.log(`Move failed: ${selectedExample.reason}`);
            }
          }
        } else if (state.turnPhase === 'draw') {
          // Draw phase - decide between deck and discard pile
          const topDiscards = state.discardPile.slice(-3).reverse();
          
          // Smart decision about where to draw from
          if (topDiscards.length > 0) {
            const visibleCard = topDiscards[0]; // The card we can see and draw
            const handBeforeDraw = [...currentPlayer.hand];
            
            // Check if the visible card helps form sets or runs
            const potentialHand = [...handBeforeDraw, visibleCard];
            const canFormSet = this.checkForSets(potentialHand, visibleCard);
            const canFormRun = this.checkForRuns(potentialHand, visibleCard);
            const cardValue = visibleCard.value;
            
            // Decide based on card utility
            const shouldDrawFromDiscard = (
              (canFormSet || canFormRun) || // Forms combinations
              (cardValue <= 3) || // Low value card
              (Math.random() < 0.3 && cardValue <= 5) // Sometimes take medium cards
            );
            
            if (shouldDrawFromDiscard) {
              const drawnCards = game.drawFromDiscard(playerId);
              actionIndex = 1; // Drew from discard
              
              reward = RewardStrategy.calculateDrawReward(
                handBeforeDraw,
                drawnCards[0],
                true, // drew from discard
                true  // could have drawn from discard
              );
              
              if (canFormSet || canFormRun) {
                console.log(`Turn ${turnCount}: ${playerId} draws ${visibleCard.rank}${visibleCard.suit} from discard (forms combination, reward: ${reward.toFixed(2)})`);
              }
            } else {
              // Draw from deck
              const drawnCards = game.drawFromDeck(playerId);
              actionIndex = 0;
              
              reward = RewardStrategy.calculateDrawReward(
                handBeforeDraw,
                drawnCards[0],
                false, // drew from deck
                true   // could have drawn from discard
              );
            }
          } else {
            // No choice, draw from deck
            game.drawFromDeck(playerId);
            actionIndex = 0;
            reward = -1;
          }
        }
        
        // Hand value change calculated in reward
        
      } catch (error) {
        // Invalid action
        reward = -20;
      }
      
      // Get next state
      const newState = game.getState();
      let nextStateTensor: tf.Tensor2D | null = null;
      
      if (newState.gamePhase === 'playing') {
        const nextPlayer = newState.players[newState.currentPlayerIndex];
        if (nextPlayer.id === playerId) {
          // Still the same player's turn (draw phase)
          nextStateTensor = this.neuralAI.encodeGameState(
            nextPlayer.hand,
            newState.discardPile,
            newState,
            nextPlayer.id
          );
        }
      }
      
      // Store experience with -Infinity handling
      // Convert -Infinity to large negative value for stability
      const safeReward = reward === -Infinity ? -1000 : reward;
      
      episodeExperiences.push({
        state: stateTensor,
        action: actionIndex,
        reward: safeReward,
        nextState: nextStateTensor,
        done: newState.gamePhase !== 'playing'
      });
    }
    
    // Log episode summary
    const winner = game.getState().players.find(p => 
      p.hand.length === 0 || game.getState().gamePhase === 'roundEnd'
    );
    console.log(`Episode ended after ${turnCount} turns. Winner: ${winner?.id || 'Unknown'}, Experiences collected: ${episodeExperiences.length}`);
    
    // Calculate returns and add experiences
    this.calculateReturns(episodeExperiences);
    this.addExperiences(episodeExperiences);
  }
  
  private checkForSets(hand: Card[], newCard: Card): boolean {
    // Check if the new card forms a set with existing cards
    const sameRankCards = hand.filter(c => c.rank === newCard.rank);
    return sameRankCards.length >= 1; // Will form at least a pair
  }
  
  private checkForRuns(hand: Card[], newCard: Card): boolean {
    // Check if the new card can form a run
    const sameSuitCards = hand.filter(c => c.suit === newCard.suit);
    if (sameSuitCards.length < 2) return false;
    
    const allCards = [...sameSuitCards, newCard].sort((a, b) => a.value - b.value);
    
    // Check for consecutive cards
    for (let i = 0; i < allCards.length - 2; i++) {
      if (allCards[i + 1].value === allCards[i].value + 1 &&
          allCards[i + 2].value === allCards[i].value + 2) {
        return true;
      }
    }
    return false;
  }
  
  private getAllValidDiscards(hand: Card[]): Card[][] {
    const validDiscards: Card[][] = [];
    
    // Single cards
    hand.forEach(card => validDiscards.push([card]));
    
    // Sets
    const cardsByRank = new Map<string, Card[]>();
    hand.forEach(card => {
      const cards = cardsByRank.get(card.rank) || [];
      cards.push(card);
      cardsByRank.set(card.rank, cards);
    });
    
    cardsByRank.forEach(cards => {
      if (cards.length >= 2) {
        for (let size = 2; size <= cards.length; size++) {
          validDiscards.push(cards.slice(0, size));
        }
      }
    });
    
    // Runs
    const cardsBySuit = new Map<string, Card[]>();
    hand.forEach(card => {
      const cards = cardsBySuit.get(card.suit) || [];
      cards.push(card);
      cardsBySuit.set(card.suit, cards);
    });
    
    cardsBySuit.forEach(cards => {
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
  }
  
  private calculateReturns(experiences: Experience[]): void {
    let runningReturn = 0;
    
    for (let i = experiences.length - 1; i >= 0; i--) {
      if (experiences[i].done) {
        runningReturn = experiences[i].reward;
      } else {
        // Handle -Infinity rewards
        if (experiences[i].reward === -Infinity) {
          runningReturn = -1000; // Convert to large negative but finite value for training
        } else {
          runningReturn = experiences[i].reward + this.gamma * runningReturn;
        }
      }
      experiences[i].reward = runningReturn; // Update with discounted return
    }
  }
  
  private addExperiences(newExperiences: Experience[]): void {
    this.experiences.push(...newExperiences);
    
    if (this.experiences.length > this.maxExperiences) {
      const toRemove = this.experiences.slice(0, this.experiences.length - this.maxExperiences);
      toRemove.forEach(exp => {
        exp.state.dispose();
        if (exp.nextState) exp.nextState.dispose();
      });
      
      this.experiences = this.experiences.slice(-this.maxExperiences);
    }
  }
  
  async train(epochs: number = 10): Promise<number> {
    if (this.experiences.length < this.batchSize) {
      console.log('Not enough experiences to train');
      return 0;
    }
    
    let totalLoss = 0;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      const batch = this.sampleBatch();
      
      const states = tf.concat(batch.map(exp => exp.state));
      const actionTargets = tf.tensor2d(
        batch.map(exp => {
          const target = new Array(55).fill(0);
          target[exp.action] = 1;
          return target;
        })
      );
      const valueTargets = tf.tensor1d(batch.map(exp => exp.reward / 100)); // Normalize
      
      const history = await this.neuralAI['model']!.fit(
        states,
        {
          action_output: actionTargets,
          value_output: valueTargets
        },
        {
          batchSize: this.batchSize,
          epochs: 1,
          verbose: 0
        }
      );
      
      totalLoss += history.history.loss[0] as number;
      
      states.dispose();
      actionTargets.dispose();
      valueTargets.dispose();
    }
    
    return totalLoss / epochs;
  }
  
  private sampleBatch(): Experience[] {
    const batch: Experience[] = [];
    const indices = new Set<number>();
    
    while (batch.length < this.batchSize && batch.length < this.experiences.length) {
      const index = Math.floor(Math.random() * this.experiences.length);
      if (!indices.has(index)) {
        indices.add(index);
        batch.push(this.experiences[index]);
      }
    }
    
    return batch;
  }
  
  async runTrainingLoop(episodes: number = 100, trainEvery: number = 10): Promise<void> {
    console.log(`Starting training with ${episodes} episodes...`);
    
    for (let i = 0; i < episodes; i++) {
      await this.runSelfPlayEpisode();
      
      if ((i + 1) % trainEvery === 0 && this.experiences.length >= this.batchSize) {
        const avgLoss = await this.train();
        console.log(`Episode ${i + 1}: Experiences=${this.experiences.length}, Avg Loss=${avgLoss.toFixed(4)}`);
      }
      
      if ((i + 1) % 50 === 0) {
        await this.neuralAI.saveModel('localstorage://yaniv-ai-model');
        console.log('Model checkpoint saved');
      }
    }
    
    // Final save
    await this.neuralAI.saveModel('localstorage://yaniv-ai-model');
    console.log('Training complete! Final model saved.');
  }
  
  getNeuralAI(): NeuralNetworkAI {
    return this.neuralAI;
  }
}