/**
 * EvolutionRLAI: The playable AI that uses the trained model from the
 * Evolution → Distillation → RL pipeline.
 *
 * This AI integrates with the existing game UI by implementing the same
 * decision interface as ComputerAI / EnhancedComputerAI.
 */

import * as tf from '@tensorflow/tfjs';
import { Card, GameState } from '../../types/game';
import { CardUtils } from '../Card';
import {
  GameSnapshot, encodeState,
  enumerateValidDiscards, cardToIndex,
} from './StateEncoder';

export class EvolutionRLAI {
  private model: tf.LayersModel | null = null;
  private temperature: number;
  private ready = false;

  constructor(temperature: number = 0.5) {
    this.temperature = temperature;
  }

  /** Load a trained model from a URL or path */
  async loadModel(path: string): Promise<boolean> {
    try {
      this.model = await tf.loadLayersModel(path);
      this.ready = true;
      return true;
    } catch (err) {
      console.error('Failed to load Evolution-RL model:', err);
      return false;
    }
  }

  /** Set the model directly (e.g., from a just-completed training run) */
  setModel(model: tf.LayersModel): void {
    this.model = model;
    this.ready = true;
  }

  isReady(): boolean {
    return this.ready;
  }

  /**
   * Make a decision given the current game state.
   * Compatible with the existing AI interface used by GameBoardAI.
   */
  async makeDecision(
    hand: Card[],
    discardPile: Card[],
    canCallYaniv: boolean,
    gameState: Partial<GameState>,
    playerId: string,
  ): Promise<{
    action: 'yaniv' | 'draw' | 'discard';
    drawSource?: 'deck' | 'discard';
    cardsToDiscard?: Card[];
  }> {
    if (!this.model) {
      // Fallback to simple rule-based
      return this.fallbackDecision(hand, discardPile, canCallYaniv);
    }

    const turnPhase = gameState.turnPhase || 'discard';

    // Build the game snapshot
    const topDiscard = discardPile.length > 0
      ? discardPile[discardPile.length - 1]
      : null;

    const opponent = gameState.players?.find(p => p.id !== playerId);

    const snapshot: GameSnapshot = {
      hand: [...hand],
      topDiscard,
      handValue: CardUtils.getHandValue(hand),
      handSize: hand.length,
      opponentHandSize: opponent?.hand.length ?? 5,
      myScore: gameState.players?.find(p => p.id === playerId)?.score ?? 0,
      opponentScore: opponent?.score ?? 0,
      canCallYaniv,
      roundNumber: gameState.roundNumber ?? 1,
      discardPileSize: discardPile.length,
      turnPhase,
    };

    if (turnPhase === 'draw') {
      return this.makeDrawDecision(snapshot);
    } else {
      return this.makeDiscardDecision(snapshot);
    }
  }

  private makeDiscardDecision(snapshot: GameSnapshot): {
    action: 'yaniv' | 'discard';
    cardsToDiscard?: Card[];
  } {
    return tf.tidy(() => {
      const state = tf.tensor2d([Array.from(encodeState(snapshot))]);
      const outputs = this.model!.predict(state) as tf.Tensor[];
      const cardLogits = outputs[0].dataSync(); // [52]
      const yanivLogit = outputs[1].dataSync(); // [1]

      const validCombos = enumerateValidDiscards(snapshot.hand);
      if (validCombos.length === 0) {
        return { action: 'discard' as const, cardsToDiscard: [snapshot.hand[0]] };
      }

      // Score each valid combo
      const scores: number[] = [];
      for (const combo of validCombos) {
        let score = 0;
        for (const card of combo) {
          score += cardLogits[cardToIndex(card)];
        }
        scores.push(score / this.temperature);
      }

      // Add Yaniv option
      if (snapshot.canCallYaniv) {
        scores.push(yanivLogit[0] / this.temperature);
      }

      // Softmax and sample
      const maxScore = Math.max(...scores);
      const expScores = scores.map(s => Math.exp(s - maxScore));
      const sumExp = expScores.reduce((a, b) => a + b, 0);
      const probs = expScores.map(e => e / sumExp);

      // Use argmax for play (deterministic) instead of sampling
      let bestIdx = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[bestIdx]) bestIdx = i;
      }

      if (snapshot.canCallYaniv && bestIdx === validCombos.length) {
        return { action: 'yaniv' as const };
      }

      return {
        action: 'discard' as const,
        cardsToDiscard: validCombos[Math.min(bestIdx, validCombos.length - 1)],
      };
    });
  }

  private makeDrawDecision(snapshot: GameSnapshot): {
    action: 'draw';
    drawSource: 'deck' | 'discard';
  } {
    const source = tf.tidy(() => {
      const state = tf.tensor2d([Array.from(encodeState(snapshot))]);
      const outputs = this.model!.predict(state) as tf.Tensor[];
      const drawLogit = outputs[2].dataSync(); // [1]

      const prob = 1 / (1 + Math.exp(-drawLogit[0]));
      // Use threshold for deterministic play
      return prob > 0.5 ? 'discard' : 'deck';
    });

    return { action: 'draw', drawSource: source as 'deck' | 'discard' };
  }

  /** Simple fallback when model isn't loaded */
  private fallbackDecision(
    hand: Card[],
    _discardPile: Card[],
    canCallYaniv: boolean,
  ): {
    action: 'yaniv' | 'draw' | 'discard';
    drawSource?: 'deck' | 'discard';
    cardsToDiscard?: Card[];
  } {
    const handValue = CardUtils.getHandValue(hand);

    if (canCallYaniv && handValue <= 5) {
      return { action: 'yaniv' };
    }

    // Discard highest value single card
    const sorted = [...hand].sort((a, b) => b.value - a.value);
    return {
      action: 'discard',
      cardsToDiscard: [sorted[0]],
    };
  }

  /** Get model info for display */
  getInfo(): { ready: boolean; temperature: number; paramCount: number } {
    let paramCount = 0;
    if (this.model) {
      for (const w of this.model.getWeights()) {
        paramCount += w.size;
      }
    }
    return {
      ready: this.ready,
      temperature: this.temperature,
      paramCount,
    };
  }
}
