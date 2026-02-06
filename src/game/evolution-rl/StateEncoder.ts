/**
 * StateEncoder: Converts Yaniv game state into a numeric feature vector
 * for neural network input. Shared between distillation and RL phases.
 *
 * Feature layout (113 dimensions):
 * [0-51]   : Hand cards (52 binary: 1 if card is in hand)
 * [52-103] : Top discard card (52 binary: one-hot encoding)
 * [104]    : Hand value / 50 (normalized)
 * [105]    : Hand size / 10 (normalized)
 * [106]    : Opponent hand size / 10 (normalized)
 * [107]    : My score / 100 (normalized)
 * [108]    : Opponent score / 100 (normalized)
 * [109]    : Can call Yaniv (binary)
 * [110]    : Round number / 20 (normalized)
 * [111]    : Discard pile size / 52 (normalized)
 * [112]    : Turn phase (0=discard, 1=draw)
 */

import { Card, Suit, Rank } from '../../types/game';
import { CardUtils } from '../Card';

export const STATE_SIZE = 113;

const SUITS: Suit[] = ['♠', '♥', '♦', '♣'];
const RANKS: Rank[] = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];

export function cardToIndex(card: Card): number {
  const suitIdx = SUITS.indexOf(card.suit);
  const rankIdx = RANKS.indexOf(card.rank);
  return suitIdx * 13 + rankIdx;
}

export function indexToCard(index: number): Card {
  const suitIdx = Math.floor(index / 13);
  const rankIdx = index % 13;
  return CardUtils.create(SUITS[suitIdx], RANKS[rankIdx]);
}

export interface GameSnapshot {
  hand: Card[];
  topDiscard: Card | null;
  handValue: number;
  handSize: number;
  opponentHandSize: number;
  myScore: number;
  opponentScore: number;
  canCallYaniv: boolean;
  roundNumber: number;
  discardPileSize: number;
  turnPhase: 'discard' | 'draw';
}

export function encodeState(snapshot: GameSnapshot): Float32Array {
  const state = new Float32Array(STATE_SIZE);

  // Hand cards: binary presence
  for (const card of snapshot.hand) {
    state[cardToIndex(card)] = 1;
  }

  // Top discard card: one-hot
  if (snapshot.topDiscard) {
    state[52 + cardToIndex(snapshot.topDiscard)] = 1;
  }

  // Normalized features
  state[104] = snapshot.handValue / 50;
  state[105] = snapshot.handSize / 10;
  state[106] = snapshot.opponentHandSize / 10;
  state[107] = snapshot.myScore / 100;
  state[108] = snapshot.opponentScore / 100;
  state[109] = snapshot.canCallYaniv ? 1 : 0;
  state[110] = snapshot.roundNumber / 20;
  state[111] = snapshot.discardPileSize / 52;
  state[112] = snapshot.turnPhase === 'draw' ? 1 : 0;

  return state;
}

/** Enumerate all valid discard combinations for a given hand */
export function enumerateValidDiscards(hand: Card[]): Card[][] {
  const combos: Card[][] = [];

  // Single cards
  for (const card of hand) {
    combos.push([card]);
  }

  // Sets (same rank, 2+)
  const byRank = new Map<string, Card[]>();
  for (const card of hand) {
    const arr = byRank.get(card.rank) || [];
    arr.push(card);
    byRank.set(card.rank, arr);
  }
  for (const cards of byRank.values()) {
    if (cards.length >= 2) {
      // All subsets of size 2+
      for (let size = 2; size <= cards.length; size++) {
        combos.push(cards.slice(0, size));
      }
    }
  }

  // Runs (same suit, 3+ consecutive)
  const bySuit = new Map<string, Card[]>();
  for (const card of hand) {
    const arr = bySuit.get(card.suit) || [];
    arr.push(card);
    bySuit.set(card.suit, arr);
  }
  for (const cards of bySuit.values()) {
    if (cards.length < 3) continue;
    const sorted = [...cards].sort((a, b) => a.value - b.value);
    for (let start = 0; start < sorted.length - 2; start++) {
      for (let end = start + 2; end < sorted.length; end++) {
        const run = sorted.slice(start, end + 1);
        if (CardUtils.isRun(run)) {
          combos.push(run);
        }
      }
    }
  }

  return combos;
}
