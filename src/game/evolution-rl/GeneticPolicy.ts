/**
 * GeneticPolicy: A parameterized heuristic policy for Yaniv that can be
 * evolved via genetic algorithms.
 *
 * 29 parameters control all aspects of play:
 * - Yaniv calling decisions (3)
 * - Card valuation for discard priority (13, one per rank)
 * - Combo preferences (4)
 * - Draw strategy (5)
 * - Strategic weights (4)
 */

import { Card } from '../../types/game';
import { CardUtils } from '../Card';
import { enumerateValidDiscards, GameSnapshot } from './StateEncoder';

export const GENOME_SIZE = 29;

export interface GeneticPolicyGenome {
  // Yaniv decision (3)
  yanivThreshold: number;       // [1, 7] hand value below which to call Yaniv
  yanivAggressiveness: number;  // [0, 1] probability of calling Yaniv even at edge
  assafRiskFactor: number;      // [0, 1] sensitivity to opponent hand size

  // Card discard valuation (13, one per rank A-K)
  // Higher = more eager to discard this rank
  cardDiscardScores: number[];  // [0, 10] each

  // Combo preferences (4)
  pairBonus: number;           // [0, 5] bonus for discarding pairs
  tripleBonus: number;         // [0, 10] bonus for triples
  runBonus: number;            // [0, 10] bonus for runs
  comboSizeWeight: number;     // [0, 2] per-extra-card bonus in combos

  // Draw strategy (5)
  drawDiscardPreference: number; // [-1, 1] base preference for discard draw
  drawLowCardBonus: number;     // [0, 3] bonus for low-value discard cards
  drawSetPotential: number;     // [0, 5] bonus when discard matches hand ranks
  drawRunPotential: number;     // [0, 5] bonus when discard extends a run
  drawValueThreshold: number;   // [1, 13] max value to consider from discard

  // Strategic (4)
  keepComboCards: number;        // [0, 2] keep cards that form combos
  handReductionUrgency: number;  // [0, 2] urgency to reduce hand value
  endgameAggression: number;     // [0, 2] aggression when scores high
  opponentPressure: number;      // [0, 2] react to small opponent hand
}

export class GeneticPolicy {
  genome: GeneticPolicyGenome;
  fitness: number = 0;

  constructor(genome?: GeneticPolicyGenome) {
    this.genome = genome || GeneticPolicy.randomGenome();
  }

  /** Create a random genome */
  static randomGenome(): GeneticPolicyGenome {
    return {
      yanivThreshold: 1 + Math.random() * 6,
      yanivAggressiveness: Math.random(),
      assafRiskFactor: Math.random(),
      cardDiscardScores: Array.from({ length: 13 }, () => Math.random() * 10),
      pairBonus: Math.random() * 5,
      tripleBonus: Math.random() * 10,
      runBonus: Math.random() * 10,
      comboSizeWeight: Math.random() * 2,
      drawDiscardPreference: Math.random() * 2 - 1,
      drawLowCardBonus: Math.random() * 3,
      drawSetPotential: Math.random() * 5,
      drawRunPotential: Math.random() * 5,
      drawValueThreshold: 1 + Math.random() * 12,
      keepComboCards: Math.random() * 2,
      handReductionUrgency: Math.random() * 2,
      endgameAggression: Math.random() * 2,
      opponentPressure: Math.random() * 2,
    };
  }

  /** Serialize genome to flat array */
  toArray(): number[] {
    const g = this.genome;
    return [
      g.yanivThreshold,
      g.yanivAggressiveness,
      g.assafRiskFactor,
      ...g.cardDiscardScores,
      g.pairBonus,
      g.tripleBonus,
      g.runBonus,
      g.comboSizeWeight,
      g.drawDiscardPreference,
      g.drawLowCardBonus,
      g.drawSetPotential,
      g.drawRunPotential,
      g.drawValueThreshold,
      g.keepComboCards,
      g.handReductionUrgency,
      g.endgameAggression,
      g.opponentPressure,
    ];
  }

  /** Deserialize genome from flat array */
  static fromArray(arr: number[]): GeneticPolicy {
    let i = 0;
    const genome: GeneticPolicyGenome = {
      yanivThreshold: arr[i++],
      yanivAggressiveness: arr[i++],
      assafRiskFactor: arr[i++],
      cardDiscardScores: arr.slice(i, i + 13),
      pairBonus: (i += 13, arr[i++]),
      tripleBonus: arr[i++],
      runBonus: arr[i++],
      comboSizeWeight: arr[i++],
      drawDiscardPreference: arr[i++],
      drawLowCardBonus: arr[i++],
      drawSetPotential: arr[i++],
      drawRunPotential: arr[i++],
      drawValueThreshold: arr[i++],
      keepComboCards: arr[i++],
      handReductionUrgency: arr[i++],
      endgameAggression: arr[i++],
      opponentPressure: arr[i++],
    };
    return new GeneticPolicy(genome);
  }

  /** Crossover: uniform crossover between two parents */
  static crossover(parent1: GeneticPolicy, parent2: GeneticPolicy): GeneticPolicy {
    const arr1 = parent1.toArray();
    const arr2 = parent2.toArray();
    const child = arr1.map((v, i) => Math.random() < 0.5 ? v : arr2[i]);
    return GeneticPolicy.fromArray(child);
  }

  /** Mutate: perturb weights with Gaussian noise */
  static mutate(
    policy: GeneticPolicy,
    mutationRate: number = 0.2,
    mutationStrength: number = 0.5,
  ): GeneticPolicy {
    const arr = policy.toArray();
    const mutated = arr.map(v => {
      if (Math.random() < mutationRate) {
        return v + (Math.random() * 2 - 1) * mutationStrength;
      }
      return v;
    });
    // Clamp values to reasonable ranges
    return GeneticPolicy.fromArray(GeneticPolicy.clampGenome(mutated));
  }

  private static clampGenome(arr: number[]): number[] {
    const clamped = [...arr];
    clamped[0] = Math.max(1, Math.min(7, clamped[0]));      // yanivThreshold
    clamped[1] = Math.max(0, Math.min(1, clamped[1]));      // yanivAggressiveness
    clamped[2] = Math.max(0, Math.min(1, clamped[2]));      // assafRiskFactor
    for (let i = 3; i < 16; i++) {                           // cardDiscardScores
      clamped[i] = Math.max(0, Math.min(10, clamped[i]));
    }
    clamped[16] = Math.max(0, Math.min(5, clamped[16]));    // pairBonus
    clamped[17] = Math.max(0, Math.min(10, clamped[17]));   // tripleBonus
    clamped[18] = Math.max(0, Math.min(10, clamped[18]));   // runBonus
    clamped[19] = Math.max(0, Math.min(2, clamped[19]));    // comboSizeWeight
    clamped[20] = Math.max(-1, Math.min(1, clamped[20]));   // drawDiscardPreference
    clamped[21] = Math.max(0, Math.min(3, clamped[21]));    // drawLowCardBonus
    clamped[22] = Math.max(0, Math.min(5, clamped[22]));    // drawSetPotential
    clamped[23] = Math.max(0, Math.min(5, clamped[23]));    // drawRunPotential
    clamped[24] = Math.max(1, Math.min(13, clamped[24]));   // drawValueThreshold
    clamped[25] = Math.max(0, Math.min(2, clamped[25]));    // keepComboCards
    clamped[26] = Math.max(0, Math.min(2, clamped[26]));    // handReductionUrgency
    clamped[27] = Math.max(0, Math.min(2, clamped[27]));    // endgameAggression
    clamped[28] = Math.max(0, Math.min(2, clamped[28]));    // opponentPressure
    return clamped;
  }

  // ═══════════════════════════════════════
  //  DECISION MAKING
  // ═══════════════════════════════════════

  /** Decide what to do during discard phase */
  decideDiscard(snapshot: GameSnapshot): { action: 'yaniv' | 'discard'; cards?: Card[] } {
    const g = this.genome;
    const { hand, opponentHandSize, opponentScore } = snapshot;
    const handValue = CardUtils.getHandValue(hand);

    // --- Yaniv decision ---
    if (snapshot.canCallYaniv && handValue <= g.yanivThreshold) {
      // Risk assessment: if opponent has very few cards, they might have lower
      const riskPenalty = g.assafRiskFactor * Math.max(0, 3 - opponentHandSize) * 0.3;
      // Endgame aggression: more willing to call when scores are high
      const endgameBonus = g.endgameAggression * (opponentScore / 100) * 0.2;
      // Pressure: opponent has few cards, they might call Yaniv soon
      const pressureBonus = g.opponentPressure * Math.max(0, 3 - opponentHandSize) * 0.15;

      const yanivProb = g.yanivAggressiveness + endgameBonus + pressureBonus - riskPenalty;
      if (Math.random() < yanivProb || handValue <= 3) {
        return { action: 'yaniv' };
      }
    }

    // --- Discard decision ---
    const validDiscards = enumerateValidDiscards(hand);
    if (validDiscards.length === 0) {
      // Should not happen, but safety
      return { action: 'discard', cards: [hand[0]] };
    }

    let bestScore = -Infinity;
    let bestDiscard: Card[] = validDiscards[0];

    for (const combo of validDiscards) {
      let score = 0;

      // Base score: sum of discard scores for each card
      for (const card of combo) {
        const rankIdx = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'].indexOf(card.rank);
        score += g.cardDiscardScores[rankIdx];
      }

      // Hand reduction urgency: prefer discards that reduce hand value more
      const valueReduction = combo.reduce((s, c) => s + c.value, 0);
      score += g.handReductionUrgency * valueReduction * 0.5;

      // Combo bonuses
      if (combo.length >= 2 && CardUtils.isSet(combo)) {
        score += combo.length === 2 ? g.pairBonus : g.tripleBonus;
      }
      if (combo.length >= 3 && CardUtils.isRun(combo)) {
        score += g.runBonus;
      }
      if (combo.length > 1) {
        score += g.comboSizeWeight * (combo.length - 1);
      }

      // Penalty for discarding cards that could form combos with remaining hand
      const remainingHand = hand.filter(
        h => !combo.some(c => c.suit === h.suit && c.rank === h.rank)
      );
      for (const card of combo) {
        // Check if this card has same-rank partners in remaining hand
        const sameRank = remainingHand.filter(c => c.rank === card.rank).length;
        if (sameRank > 0) {
          score -= g.keepComboCards * sameRank * 0.5;
        }
        // Check if this card is part of a potential run
        const sameSuit = remainingHand.filter(c => c.suit === card.suit);
        for (const c of sameSuit) {
          if (Math.abs(c.value - card.value) <= 2) {
            score -= g.keepComboCards * 0.3;
          }
        }
      }

      if (score > bestScore) {
        bestScore = score;
        bestDiscard = combo;
      }
    }

    return { action: 'discard', cards: bestDiscard };
  }

  /** Decide where to draw from during draw phase */
  decideDraw(snapshot: GameSnapshot): 'deck' | 'discard' {
    const g = this.genome;
    const { hand, topDiscard } = snapshot;

    if (!topDiscard) return 'deck';

    let discardScore = g.drawDiscardPreference;

    // Low card bonus
    if (topDiscard.value <= 5) {
      discardScore += g.drawLowCardBonus * (6 - topDiscard.value) / 5;
    }

    // Value threshold: don't draw high cards
    if (topDiscard.value > g.drawValueThreshold) {
      discardScore -= 3;
    }

    // Set potential: discard card matches a rank in hand
    const sameRank = hand.filter(c => c.rank === topDiscard.rank).length;
    if (sameRank >= 1) {
      discardScore += g.drawSetPotential * sameRank;
    }

    // Run potential: discard card extends a run in hand
    const sameSuit = hand.filter(c => c.suit === topDiscard.suit);
    let runExtension = 0;
    for (const card of sameSuit) {
      if (Math.abs(card.value - topDiscard.value) === 1) {
        runExtension++;
      }
    }
    if (runExtension > 0) {
      discardScore += g.drawRunPotential * runExtension * 0.5;
    }

    return discardScore > 0 ? 'discard' : 'deck';
  }
}
