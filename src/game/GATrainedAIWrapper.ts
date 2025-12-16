/**
 * GA-Trained AI Wrapper
 *
 * Wraps GATrainedAI to match the existing AI interface used by GameBoard components.
 */

import { Card, GameState } from '../types/game';
import { CardUtils } from './Card';

// Import types from GATrainedAI
interface YanivAction {
  type: 'discard' | 'yaniv';
  discardCards?: Card[];
  drawSource?: 'deck' | 'discard';
}

interface ModelConfig {
  version: string;
  architecture: {
    type: string;
    input_dim: number;
    hidden_dim: number;
    hidden_dim2: number;
    output_dim: number;
  };
  encoding: {
    state_dim: number;
    action_dim: number;
    num_cards: number;
    suit_order: string[];
    rank_order: string[];
  };
  weights: {
    layer1: { weight: number[][]; bias: number[] };
    layer2: { weight: number[][]; bias: number[] };
    layer3: { weight: number[][]; bias: number[] };
  };
  metadata?: {
    genome_id: string;
    fitness: number;
    generation: number;
    wins: number;
    games_played: number;
  };
}

export class GATrainedAIWrapper {
  private config: ModelConfig | null = null;
  private suitOrder: string[] = [];
  private rankOrder: string[] = [];
  private modelLoaded: boolean = false;

  async loadModel(modelPath: string): Promise<void> {
    try {
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error(`Failed to load model: ${response.statusText}`);
      }
      this.config = await response.json();
      this.suitOrder = this.config!.encoding.suit_order;
      this.rankOrder = this.config!.encoding.rank_order;
      this.modelLoaded = true;
      console.log('GA-trained model loaded:', this.config?.metadata);
    } catch (error) {
      console.error('Error loading GA model:', error);
      throw error;
    }
  }

  isLoaded(): boolean {
    return this.modelLoaded;
  }

  /**
   * Make a move decision - matches PythonTrainedAI interface
   */
  makeMove(
    gameState: GameState,
    playerIndex: number
  ): {
    action: 'yaniv' | 'discard' | 'draw';
    cards?: Card[];
    source?: 'deck' | 'discard';
  } {
    if (!this.config || !this.modelLoaded) {
      console.warn('Model not loaded, using fallback');
      return { action: 'draw', source: 'deck' };
    }

    const player = gameState.players[playerIndex];
    const hand = player.hand;
    const discardPile = gameState.discardPile;
    const turnPhase = gameState.turnPhase;

    // Get opponent info
    const opponents = gameState.players
      .filter((_, i) => i !== playerIndex)
      .map(p => ({
        handCount: p.hand.length,
        score: p.score
      }));

    // Check if can call yaniv
    const handValue = CardUtils.getHandValue(hand);
    const canCallYaniv = handValue <= 7 && !player.hasCalledYanivLastRound;

    // Get drawable discard card
    const drawableDiscard = discardPile.length > 0 ? discardPile[discardPile.length - 1] : null;

    // If in draw phase, only return draw decision
    if (turnPhase === 'draw') {
      const action = this.chooseAction(
        hand,
        discardPile,
        opponents,
        gameState.roundNumber,
        false, // Can't call yaniv during draw phase
        drawableDiscard
      );
      return {
        action: 'draw',
        source: action.drawSource || 'deck'
      };
    }

    // Discard phase - get full action
    const action = this.chooseAction(
      hand,
      discardPile,
      opponents,
      gameState.roundNumber,
      canCallYaniv,
      drawableDiscard
    );

    if (action.type === 'yaniv') {
      return { action: 'yaniv' };
    }

    return {
      action: 'discard',
      cards: action.discardCards,
      source: action.drawSource
    };
  }

  /**
   * Choose best action using the neural network
   */
  private chooseAction(
    hand: Card[],
    discardPile: Card[],
    opponents: { handCount: number; score: number }[],
    roundNumber: number,
    canCallYaniv: boolean,
    drawableDiscard: Card | null
  ): YanivAction {
    const handValue = CardUtils.getHandValue(hand);
    const canDrawFromDiscard = drawableDiscard !== null;

    // Generate legal actions
    const legalActions = this.generateLegalActions(hand, canCallYaniv, canDrawFromDiscard);

    if (legalActions.length === 0) {
      // Fallback
      return { type: 'discard', discardCards: [hand[0]], drawSource: 'deck' };
    }

    if (legalActions.length === 1) {
      return legalActions[0];
    }

    // Encode state
    const stateVec = this.encodeState(
      hand, discardPile, handValue, opponents, roundNumber, canCallYaniv
    );

    // Score each action
    let bestAction = legalActions[0];
    let bestScore = -Infinity;

    for (const action of legalActions) {
      const actionVec = this.encodeAction(action, hand, drawableDiscard);
      const input = [...stateVec, ...actionVec];
      const score = this.forward(input);

      if (score > bestScore) {
        bestScore = score;
        bestAction = action;
      }
    }

    return bestAction;
  }

  private forward(input: number[]): number {
    if (!this.config) return 0;

    const weights = this.config.weights;

    // Layer 1
    let h1 = this.matmul(input, weights.layer1.weight);
    h1 = this.addBias(h1, weights.layer1.bias);
    h1 = this.relu(h1);

    // Layer 2
    let h2 = this.matmul(h1, weights.layer2.weight);
    h2 = this.addBias(h2, weights.layer2.bias);
    h2 = this.relu(h2);

    // Layer 3
    let out = this.matmul(h2, weights.layer3.weight);
    out = this.addBias(out, weights.layer3.bias);

    return out[0];
  }

  private matmul(vec: number[], mat: number[][]): number[] {
    const result: number[] = [];
    for (let j = 0; j < mat[0].length; j++) {
      let sum = 0;
      for (let i = 0; i < vec.length; i++) {
        sum += vec[i] * mat[i][j];
      }
      result.push(sum);
    }
    return result;
  }

  private addBias(vec: number[], bias: number[]): number[] {
    return vec.map((v, i) => v + bias[i]);
  }

  private relu(vec: number[]): number[] {
    return vec.map(v => Math.max(0, v));
  }

  private cardToIndex(card: Card): number {
    const suitIdx = this.suitOrder.indexOf(card.suit);
    const rankIdx = this.rankOrder.indexOf(card.rank);
    return suitIdx * 13 + rankIdx;
  }

  private encodeState(
    hand: Card[],
    discardPile: Card[],
    handValue: number,
    opponents: { handCount: number; score: number }[],
    roundNumber: number,
    canCallYaniv: boolean
  ): number[] {
    const features: number[] = [];

    // Hand encoding (52-dim)
    const handEncoding = new Array(52).fill(0);
    for (const card of hand) {
      handEncoding[this.cardToIndex(card)] = 1;
    }
    features.push(...handEncoding);

    // Top 3 discards (156-dim)
    const discardEncoding = new Array(156).fill(0);
    const topDiscards = discardPile.slice(-3).reverse();
    for (let i = 0; i < topDiscards.length && i < 3; i++) {
      const idx = i * 52 + this.cardToIndex(topDiscards[i]);
      discardEncoding[idx] = 1;
    }
    features.push(...discardEncoding);

    // Scalars (5-dim)
    features.push(handValue / 100);
    features.push(hand.length / 10);
    features.push(discardPile.length / 52);
    features.push(roundNumber / 20);
    features.push(canCallYaniv ? 1 : 0);

    // Opponent features (6-dim)
    for (let i = 0; i < 3; i++) {
      if (i < opponents.length) {
        features.push(opponents[i].handCount / 10);
        features.push(opponents[i].score / 200);
      } else {
        features.push(0);
        features.push(0);
      }
    }

    return features;
  }

  private encodeAction(
    action: YanivAction,
    hand: Card[],
    drawableDiscard: Card | null
  ): number[] {
    const features: number[] = [];

    // Action type (4-dim)
    const actionType = new Array(4).fill(0);
    if (action.type === 'yaniv') {
      actionType[3] = 1;
    } else {
      actionType[0] = 1;
    }
    features.push(...actionType);

    // Draw source (2-dim)
    const drawSource = new Array(2).fill(0);
    if (action.drawSource === 'discard') {
      drawSource[1] = 1;
    } else {
      drawSource[0] = 1;
    }
    features.push(...drawSource);

    // Discard mask (52-dim)
    const discardMask = new Array(52).fill(0);
    if (action.discardCards) {
      for (const card of action.discardCards) {
        discardMask[this.cardToIndex(card)] = 1;
      }
    }
    features.push(...discardMask);

    // Scalars (4-dim)
    const discardCards = action.discardCards || [];
    const discardCount = discardCards.length;
    const discardValue = discardCards.reduce((sum, c) => sum + c.value, 0);

    let resultValue: number;
    if (action.type === 'yaniv') {
      resultValue = CardUtils.getHandValue(hand);
    } else {
      const remaining = hand.filter(c =>
        !discardCards.some(dc => CardUtils.areEqual(dc, c))
      );
      const drawValue = action.drawSource === 'discard' && drawableDiscard
        ? drawableDiscard.value
        : 7;
      resultValue = CardUtils.getHandValue(remaining) + drawValue;
    }

    features.push(discardCount / 5);
    features.push(discardValue / 50);
    features.push(resultValue / 100);
    features.push(action.type === 'yaniv' ? 1 : 0);

    return features;
  }

  private generateLegalActions(
    hand: Card[],
    canCallYaniv: boolean,
    canDrawFromDiscard: boolean
  ): YanivAction[] {
    const actions: YanivAction[] = [];

    if (canCallYaniv) {
      actions.push({ type: 'yaniv' });
    }

    const discardCombos = this.getValidDiscardCombinations(hand);

    for (const combo of discardCombos) {
      actions.push({
        type: 'discard',
        discardCards: combo,
        drawSource: 'deck'
      });

      if (canDrawFromDiscard) {
        actions.push({
          type: 'discard',
          discardCards: combo,
          drawSource: 'discard'
        });
      }
    }

    return actions;
  }

  private getValidDiscardCombinations(hand: Card[]): Card[][] {
    const combos: Card[][] = [];

    // Singles
    for (const card of hand) {
      combos.push([card]);
    }

    // Sets
    const byRank = new Map<string, Card[]>();
    for (const card of hand) {
      const existing = byRank.get(card.rank) || [];
      existing.push(card);
      byRank.set(card.rank, existing);
    }

    for (const cards of byRank.values()) {
      if (cards.length >= 2) {
        for (let size = 2; size <= cards.length; size++) {
          const subsets = this.getCombinations(cards, size);
          combos.push(...subsets);
        }
      }
    }

    // Runs
    const bySuit = new Map<string, Card[]>();
    for (const card of hand) {
      const existing = bySuit.get(card.suit) || [];
      existing.push(card);
      bySuit.set(card.suit, existing);
    }

    for (const cards of bySuit.values()) {
      if (cards.length >= 3) {
        const sorted = [...cards].sort((a, b) => a.value - b.value);
        for (let start = 0; start < sorted.length; start++) {
          const run: Card[] = [sorted[start]];
          for (let next = start + 1; next < sorted.length; next++) {
            if (sorted[next].value === run[run.length - 1].value + 1) {
              run.push(sorted[next]);
              if (run.length >= 3) {
                combos.push([...run]);
              }
            } else {
              break;
            }
          }
        }
      }
    }

    return combos;
  }

  private getCombinations<T>(arr: T[], size: number): T[][] {
    if (size === 0) return [[]];
    if (arr.length < size) return [];

    const result: T[][] = [];
    for (let i = 0; i <= arr.length - size; i++) {
      const rest = this.getCombinations(arr.slice(i + 1), size - 1);
      for (const combo of rest) {
        result.push([arr[i], ...combo]);
      }
    }
    return result;
  }
}
