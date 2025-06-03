import { Card, GameState } from '../types/game';
import { CardUtils } from './Card';

interface PythonModel {
  network_id: number;
  w1: number[][];  // 11x64
  b1: number[];    // 64
  w2: number[][];  // 64x16
  b2: number[];    // 16
  fitness: number;
  wins: number;
  games_played: number;
}

export class PythonTrainedAI {
  private model: PythonModel | null = null;
  private readonly inputSize = 11;
  private readonly hiddenSize = 64;
  private readonly outputSize = 16;

  async loadModel(modelPath: string = '/saved_networks/best_overall_optimized.json') {
    try {
      const response = await fetch(modelPath);
      this.model = await response.json();
      console.log(`Loaded AI model with fitness: ${this.model?.fitness.toFixed(3)}`);
      console.log(`Win rate: ${this.model?.wins}/${this.model?.games_played} games`);
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  private encodeState(gameState: GameState, playerIndex: number): number[] {
    const features = new Array(this.inputSize).fill(0);
    const player = gameState.players[playerIndex];
    
    // Features 0-4: Hand cards (up to 5, normalized by /13)
    for (let i = 0; i < Math.min(5, player.hand.length); i++) {
      features[i] = player.hand[i].value / 13.0;
    }
    
    // Feature 5: Hand value (normalized by /50)
    features[5] = CardUtils.getHandValue(player.hand) / 50.0;
    
    // Feature 6: Deck size (normalized by /54)
    features[6] = gameState.deck.length / 54.0;
    
    // Features 7-9: Opponent card counts (normalized by /10)
    let opponentIdx = 7;
    for (let i = 0; i < gameState.players.length; i++) {
      if (i !== playerIndex && opponentIdx < 10) {
        features[opponentIdx++] = gameState.players[i].hand.length / 10.0;
      }
    }
    
    // Feature 10: Last discarded card value (normalized by /13)
    if (gameState.discardPile.length > 0) {
      const topCard = gameState.discardPile[gameState.discardPile.length - 1];
      features[10] = topCard.value / 13.0;
    }
    
    return features;
  }

  private relu(x: number): number {
    return Math.max(0, x);
  }

  private softmax(x: number[]): number[] {
    const max = Math.max(...x);
    const exp = x.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / sum);
  }

  private forward(input: number[]): number[] {
    if (!this.model) throw new Error('Model not loaded');
    
    // First layer (input -> hidden)
    const hidden = new Array(this.hiddenSize).fill(0);
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = this.model.b1[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += input[j] * this.model.w1[j][i];
      }
      hidden[i] = this.relu(sum);
    }
    
    // Second layer (hidden -> output)
    const output = new Array(this.outputSize).fill(0);
    for (let i = 0; i < this.outputSize; i++) {
      let sum = this.model.b2[i];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hidden[j] * this.model.w2[j][i];
      }
      output[i] = sum;
    }
    
    return this.softmax(output);
  }

  makeMove(gameState: GameState, playerIndex: number): {
    action: 'draw' | 'discard' | 'yaniv';
    cards?: Card[];
    source?: 'deck' | 'discard';
  } {
    if (!this.model) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    const state = this.encodeState(gameState, playerIndex);
    const actionProbs = this.forward(state);
    
    // Sample action based on probabilities
    const rand = Math.random();
    let cumProb = 0;
    let selectedAction = 0;
    
    for (let i = 0; i < actionProbs.length; i++) {
      cumProb += actionProbs[i];
      if (rand < cumProb) {
        selectedAction = i;
        break;
      }
    }
    
    // Decode action
    const player = gameState.players[playerIndex];
    
    // Check if we should call Yaniv (action 15)
    if (selectedAction === 15) {
      const handValue = CardUtils.getHandValue(player.hand);
      if (handValue <= 5) {
        return { action: 'yaniv' };
      }
    }
    
    // If we need to draw (or Yaniv was invalid)
    if (gameState.turnPhase === 'draw') {
      // For simplicity, always draw from deck
      return { action: 'draw', source: 'deck' };
    }
    
    // Discard phase - select cards to discard
    if (selectedAction < 5 && selectedAction < player.hand.length) {
      // Discard single card
      return {
        action: 'discard',
        cards: [player.hand[selectedAction]]
      };
    }
    
    // Try to find valid sets/runs
    const validSets = this.findValidSets(player.hand);
    if (validSets.length > 0) {
      // Pick a random valid set
      const setIndex = Math.floor(Math.random() * validSets.length);
      return {
        action: 'discard',
        cards: validSets[setIndex]
      };
    }
    
    // Fallback: discard highest value card
    let highestCard = player.hand[0];
    for (const card of player.hand) {
      if (card.value > highestCard.value) {
        highestCard = card;
      }
    }
    
    return {
      action: 'discard',
      cards: [highestCard]
    };
  }

  private findValidSets(hand: Card[]): Card[][] {
    const validSets: Card[][] = [];
    
    // Check for sets of same rank
    const rankGroups: Map<string, Card[]> = new Map();
    for (const card of hand) {
      const rank = card.rank;
      if (!rankGroups.has(rank)) {
        rankGroups.set(rank, []);
      }
      rankGroups.get(rank)!.push(card);
    }
    
    rankGroups.forEach((cards, _) => {
      if (cards.length >= 3) {
        validSets.push(cards.slice(0, 3));
        if (cards.length >= 4) {
          validSets.push(cards);
        }
      }
    });
    
    // Check for runs (simplified - same suit, consecutive ranks)
    const suitGroups: Map<string, Card[]> = new Map();
    for (const card of hand) {
      if (!suitGroups.has(card.suit)) {
        suitGroups.set(card.suit, []);
      }
      suitGroups.get(card.suit)!.push(card);
    }
    
    const rankOrder = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
    
    suitGroups.forEach((cards, _) => {
      if (cards.length >= 3) {
        // Sort by rank
        cards.sort((a, b) => rankOrder.indexOf(a.rank) - rankOrder.indexOf(b.rank));
        
        // Look for consecutive runs
        for (let i = 0; i <= cards.length - 3; i++) {
          const run = [cards[i]];
          let currentRankIdx = rankOrder.indexOf(cards[i].rank);
          
          for (let j = i + 1; j < cards.length; j++) {
            const nextRankIdx = rankOrder.indexOf(cards[j].rank);
            if (nextRankIdx === currentRankIdx + 1) {
              run.push(cards[j]);
              currentRankIdx = nextRankIdx;
              
              if (run.length >= 3) {
                validSets.push([...run]);
              }
            } else {
              break;
            }
          }
        }
      }
    });
    
    return validSets;
  }
}