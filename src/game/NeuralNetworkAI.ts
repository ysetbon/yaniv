import * as tf from '@tensorflow/tfjs';
import { Card, GameState } from '../types/game';
import { CardUtils } from './Card';

export class NeuralNetworkAI {
  private model: tf.LayersModel | null = null;
  private readonly stateSize = 234; // Size of encoded state
  private readonly actionSize = 55; // Possible actions (draw from deck, draw 1-3 from discard, discard combinations, yaniv)
  
  constructor() {
    console.log('Building neural network model...');
    this.buildModel();
    console.log('Neural network model built');
  }

  private buildModel() {
    const input = tf.input({ shape: [this.stateSize] });
    
    // Hidden layers with batch normalization and dropout for generalization
    let x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
    x = tf.layers.dropout({ rate: 0.3 }).apply(x) as tf.SymbolicTensor;
    
    x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.batchNormalization().apply(x) as tf.SymbolicTensor;
    x = tf.layers.dropout({ rate: 0.2 }).apply(x) as tf.SymbolicTensor;
    
    x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
    
    // Output layer - action probabilities
    const actionOutput = tf.layers.dense({ 
      units: this.actionSize, 
      activation: 'softmax',
      name: 'action_output'
    }).apply(x) as tf.SymbolicTensor;
    
    // Value head for state evaluation
    const valueOutput = tf.layers.dense({ 
      units: 1, 
      activation: 'tanh',
      name: 'value_output'
    }).apply(x) as tf.SymbolicTensor;
    
    this.model = tf.model({ 
      inputs: input, 
      outputs: [actionOutput, valueOutput]
    });
    
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: {
        action_output: 'categoricalCrossentropy',
        value_output: 'meanSquaredError'
      }
    });
  }

  encodeGameState(
    hand: Card[], 
    discardPile: Card[], 
    gameState: Partial<GameState>,
    playerId: string
  ): tf.Tensor2D {
    const encoded = new Float32Array(this.stateSize);
    let index = 0;
    
    // Encode hand (52 cards, one-hot encoding)
    for (const card of hand) {
      const cardIndex = this.cardToIndex(card);
      encoded[index + cardIndex] = 1;
    }
    index += 52;
    
    // Encode top 3 discard pile cards with emphasis on the most recent (opponent's discard)
    const topDiscards = discardPile.slice(-3).reverse();
    for (let i = 0; i < 3; i++) {
      if (i < topDiscards.length) {
        const cardIndex = this.cardToIndex(topDiscards[i]);
        // Give more weight to the most recent card (the one visible to draw)
        encoded[index + cardIndex] = i === 0 ? 1.5 : 1;
      }
      index += 52;
    }
    
    // Encode game features
    encoded[index++] = CardUtils.getHandValue(hand) / 100; // Normalized hand value
    encoded[index++] = hand.length / 10; // Normalized hand size
    encoded[index++] = discardPile.length / 52; // Normalized discard pile size
    
    // Encode opponent information (assuming 2-4 players)
    if (gameState.players) {
      const opponents = gameState.players.filter(p => p.id !== playerId);
      for (let i = 0; i < 3; i++) {
        if (i < opponents.length) {
          encoded[index++] = opponents[i].hand.length / 10;
          encoded[index++] = opponents[i].score / 500; // Normalized score
        } else {
          encoded[index++] = 0;
          encoded[index++] = 0;
        }
      }
    }
    
    // Encode round number
    if (gameState.roundNumber) {
      encoded[index++] = gameState.roundNumber / 20; // Normalized
    }
    
    // Can call yaniv
    const handValue = CardUtils.getHandValue(hand);
    encoded[index++] = handValue <= 7 ? 1 : 0;
    
    // Possible valid discards count
    const validDiscards = this.getAllValidDiscards(hand);
    encoded[index++] = validDiscards.length / 50; // Normalized
    
    return tf.tensor2d([Array.from(encoded)]);
  }

  private cardToIndex(card: Card): number {
    const suitIndex = ['♠', '♥', '♦', '♣'].indexOf(card.suit);
    const rankIndex = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'].indexOf(card.rank);
    return suitIndex * 13 + rankIndex;
  }

  private getAllValidDiscards(hand: Card[]): Card[][] {
    const validDiscards: Card[][] = [];
    
    // Single cards
    for (const card of hand) {
      validDiscards.push([card]);
    }
    
    // Sets (same rank)
    const cardsByRank = new Map<string, Card[]>();
    for (const card of hand) {
      const existing = cardsByRank.get(card.rank) || [];
      existing.push(card);
      cardsByRank.set(card.rank, existing);
    }
    
    for (const [_, cards] of cardsByRank) {
      if (cards.length >= 2) {
        for (let size = 2; size <= cards.length; size++) {
          validDiscards.push(cards.slice(0, size));
        }
      }
    }
    
    // Runs (same suit, consecutive ranks)
    const cardsBySuit = new Map<string, Card[]>();
    for (const card of hand) {
      const existing = cardsBySuit.get(card.suit) || [];
      existing.push(card);
      cardsBySuit.set(card.suit, existing);
    }
    
    for (const [_, cards] of cardsBySuit) {
      const sorted = cards.sort((a, b) => a.value - b.value);
      for (let start = 0; start < sorted.length - 2; start++) {
        for (let end = start + 2; end < sorted.length; end++) {
          const potentialRun = sorted.slice(start, end + 1);
          if (CardUtils.isRun(potentialRun)) {
            validDiscards.push(potentialRun);
          }
        }
      }
    }
    
    return validDiscards;
  }

  async makeDecision(
    hand: Card[], 
    discardPile: Card[],
    gameState: Partial<GameState>,
    playerId: string
  ): Promise<{
    action: 'yaniv' | 'draw' | 'discard';
    drawSource?: 'deck' | 'discard';
    drawCount?: number;
    cardsToDiscard?: Card[];
  }> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }
    
    // Determine current turn phase
    const turnPhase = gameState.turnPhase || 'discard';
    
    const stateTensor = this.encodeGameState(hand, discardPile, gameState, playerId);
    const [actionProbs, stateValue] = this.model.predict(stateTensor) as [tf.Tensor, tf.Tensor];
    
    // Get action probabilities
    const probs = await actionProbs.data();
    const validActions = this.getValidActionMask(hand, discardPile, turnPhase);
    
    // Apply mask to invalid actions
    const maskedProbs = Array.from(probs).map((p, i) => validActions[i] ? p : 0);
    
    // Check if any valid actions exist
    const hasValidAction = maskedProbs.some(p => p > 0);
    if (!hasValidAction) {
      console.error('No valid actions available!', { turnPhase, validActions });
      // Fallback based on phase
      if (turnPhase === 'draw') {
        return { action: 'draw', drawSource: 'deck' };
      } else {
        // Find any valid discard
        const validDiscards = this.getAllValidDiscards(hand);
        if (validDiscards.length > 0) {
          return { action: 'discard', cardsToDiscard: validDiscards[0] };
        }
      }
    }
    
    // Sample action based on probabilities (exploration)
    const actionIndex = this.sampleAction(maskedProbs);
    
    // Cleanup tensors
    stateTensor.dispose();
    actionProbs.dispose();
    stateValue.dispose();
    
    const decision = this.decodeAction(actionIndex, hand);
    
    // Validate the decision before returning
    if (!this.isValidDecision(decision, hand, discardPile, turnPhase)) {
      console.error('Invalid decision generated:', decision);
      // Return safe fallback
      if (turnPhase === 'draw') {
        return { action: 'draw', drawSource: 'deck' };
      } else {
        const validDiscards = this.getAllValidDiscards(hand);
        return { action: 'discard', cardsToDiscard: validDiscards[0] || [hand[0]] };
      }
    }
    
    return decision;
  }

  private getValidActionMask(hand: Card[], discardPile: Card[], turnPhase: 'draw' | 'discard'): boolean[] {
    const mask = new Array(this.actionSize).fill(false);
    
    if (turnPhase === 'draw') {
      // Only draw actions are valid during draw phase
      // Action 0: Draw from deck (always valid in draw phase)
      mask[0] = true;
      
      // Action 1: Draw from discard (only if cards available)
      if (discardPile.length > 0) {
        mask[1] = true;
      }
    } else if (turnPhase === 'discard') {
      // Only discard and yaniv actions are valid during discard phase
      
      // Action 4: Call Yaniv (only if hand value <= 7)
      const handValue = CardUtils.getHandValue(hand);
      if (handValue <= 7) {
        mask[4] = true;
      }
      
      // Actions 5+: Valid discards
      const validDiscards = this.getAllValidDiscards(hand);
      for (let i = 0; i < Math.min(validDiscards.length, 50); i++) {
        mask[i + 5] = true;
      }
    }
    
    return mask;
  }

  private sampleAction(probabilities: number[]): number {
    const sum = probabilities.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0; // Default to draw from deck
    
    const normalized = probabilities.map(p => p / sum);
    const random = Math.random();
    
    let cumulative = 0;
    for (let i = 0; i < normalized.length; i++) {
      cumulative += normalized[i];
      if (random < cumulative) {
        return i;
      }
    }
    
    return 0;
  }

  private decodeAction(
    actionIndex: number, 
    hand: Card[]
  ): {
    action: 'yaniv' | 'draw' | 'discard';
    drawSource?: 'deck' | 'discard';
    drawCount?: number;
    cardsToDiscard?: Card[];
  } {
    if (actionIndex === 0) {
      return { action: 'draw', drawSource: 'deck' };
    }
    
    if (actionIndex >= 1 && actionIndex <= 3) {
      return { 
        action: 'draw', 
        drawSource: 'discard', 
        drawCount: actionIndex 
      };
    }
    
    if (actionIndex === 4) {
      return { action: 'yaniv' };
    }
    
    // Discard action
    const validDiscards = this.getAllValidDiscards(hand);
    const discardIndex = actionIndex - 5;
    
    if (discardIndex < validDiscards.length) {
      return {
        action: 'discard',
        cardsToDiscard: validDiscards[discardIndex]
      };
    }
    
    // Fallback
    return { action: 'draw', drawSource: 'deck' };
  }

  async saveModel(path: string) {
    if (this.model) {
      await this.model.save(path);
    }
  }

  private isValidDecision(
    decision: any,
    hand: Card[],
    discardPile: Card[],
    turnPhase: 'draw' | 'discard'
  ): boolean {
    if (turnPhase === 'draw') {
      // Must be a draw action
      if (decision.action !== 'draw') return false;
      
      // Check draw source validity
      if (decision.drawSource === 'discard' && discardPile.length === 0) return false;
      
      return true;
    } else if (turnPhase === 'discard') {
      // Must be discard or yaniv
      if (decision.action === 'draw') return false;
      
      if (decision.action === 'yaniv') {
        const handValue = CardUtils.getHandValue(hand);
        return handValue <= 7;
      }
      
      if (decision.action === 'discard') {
        if (!decision.cardsToDiscard || decision.cardsToDiscard.length === 0) return false;
        
        // Check if cards are in hand
        for (const card of decision.cardsToDiscard) {
          if (!hand.some(h => h.suit === card.suit && h.rank === card.rank)) {
            return false;
          }
        }
        
        // Check if it's a valid discard combination
        return CardUtils.isValidDiscard(decision.cardsToDiscard);
      }
    }
    
    return false;
  }
  
  async loadModel(path: string) {
    try {
      this.model = await tf.loadLayersModel(path);
      console.log('Neural network model loaded successfully from:', path);
      return true;
    } catch (error) {
      console.error('Failed to load neural network model:', error);
      // Model remains null, will use fallback
      return false;
    }
  }
}