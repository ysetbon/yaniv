/**
 * GameSimulator: Fast headless Yaniv game simulation for training.
 * Plays complete games between two policy interfaces and optionally
 * records (state, action) trajectories for distillation.
 */

import { Card } from '../../types/game';
import { CardUtils } from '../Card';
import { Deck } from '../Deck';
import { GameSnapshot, encodeState, enumerateValidDiscards, cardToIndex } from './StateEncoder';

/** A recorded step from a game for training data */
export interface TrajectoryStep {
  state: Float32Array;
  snapshot: GameSnapshot;
  action: RecordedAction;
  reward: number;
}

export interface RecordedAction {
  phase: 'discard' | 'draw';
  type: 'yaniv' | 'discard' | 'draw_deck' | 'draw_discard';
  cards?: Card[];       // cards discarded (for discard actions)
  cardsMask?: boolean[]; // 52-dim mask of which cards were discarded
}

/** Interface any policy must implement to play in simulation */
export interface SimPolicy {
  decideDiscard(snapshot: GameSnapshot): { action: 'yaniv' | 'discard'; cards?: Card[] };
  decideDraw(snapshot: GameSnapshot): 'deck' | 'discard';
}

export interface GameResult {
  winner: 0 | 1;
  scores: [number, number];
  rounds: number;
  trajectory0: TrajectoryStep[];
  trajectory1: TrajectoryStep[];
}

interface SimPlayer {
  hand: Card[];
  score: number;
  hasCalledYanivLastRound: boolean;
}

const MAX_ROUNDS = 50;
const MAX_TURNS_PER_ROUND = 100;

export class GameSimulator {
  /**
   * Play a complete game (to 101 points) between two policies.
   * Returns the winner and optionally the trajectories.
   */
  static playGame(
    policy0: SimPolicy,
    policy1: SimPolicy,
    recordTrajectory: boolean = false,
  ): GameResult {
    const players: [SimPlayer, SimPlayer] = [
      { hand: [], score: 0, hasCalledYanivLastRound: false },
      { hand: [], score: 0, hasCalledYanivLastRound: false },
    ];
    const policies = [policy0, policy1];
    const trajectory0: TrajectoryStep[] = [];
    const trajectory1: TrajectoryStep[] = [];
    const trajectories = [trajectory0, trajectory1];

    let roundNum = 0;

    while (roundNum < MAX_ROUNDS) {
      roundNum++;
      const deck = new Deck();

      // Deal 5 cards to each player
      players[0].hand = deck.draw(5);
      players[1].hand = deck.draw(5);

      // Initial discard
      const discardPile: Card[] = deck.draw(1);

      let currentPlayer = 0;
      let turnCount = 0;
      let roundOver = false;

      while (!roundOver && turnCount < MAX_TURNS_PER_ROUND) {
        turnCount++;
        const me = players[currentPlayer];
        const opp = players[1 - currentPlayer];
        const handValue = CardUtils.getHandValue(me.hand);
        const canYaniv = handValue <= 7 && !me.hasCalledYanivLastRound;
        const topDiscard = discardPile.length > 0
          ? discardPile[discardPile.length - 1]
          : null;

        // --- Discard Phase ---
        const discardSnapshot: GameSnapshot = {
          hand: [...me.hand],
          topDiscard,
          handValue,
          handSize: me.hand.length,
          opponentHandSize: opp.hand.length,
          myScore: me.score,
          opponentScore: opp.score,
          canCallYaniv: canYaniv,
          roundNumber: roundNum,
          discardPileSize: discardPile.length,
          turnPhase: 'discard',
        };

        const discardDecision = policies[currentPlayer].decideDiscard(discardSnapshot);

        if (discardDecision.action === 'yaniv' && canYaniv) {
          // Record yaniv action
          if (recordTrajectory) {
            const cardsMask = new Array(52).fill(false);
            trajectories[currentPlayer].push({
              state: encodeState(discardSnapshot),
              snapshot: discardSnapshot,
              action: { phase: 'discard', type: 'yaniv', cardsMask },
              reward: 0, // filled in later
            });
          }

          // Resolve Yaniv
          const callerValue = CardUtils.getHandValue(me.hand);
          const oppValue = CardUtils.getHandValue(opp.hand);
          const assaf = oppValue <= callerValue;

          if (assaf) {
            me.score += 30;
            me.hasCalledYanivLastRound = true;
            opp.hasCalledYanivLastRound = false;
          } else {
            opp.score += oppValue;
            me.hasCalledYanivLastRound = false;
            opp.hasCalledYanivLastRound = false;
          }

          // 50-point reduction rule
          for (const p of players) {
            if (p.score > 0 && p.score % 50 === 0) {
              p.score -= 50;
            }
          }

          // Assign round rewards to trajectory
          if (recordTrajectory) {
            const reward0 = assaf
              ? (currentPlayer === 0 ? -1 : 0.5)
              : (currentPlayer === 0 ? 1 : -0.5);
            const reward1 = -reward0;
            for (const step of trajectory0) {
              if (step.reward === 0) step.reward = reward0 * 0.1; // small per-step
            }
            // Overwrite last step with bigger reward
            if (trajectory0.length > 0) {
              trajectory0[trajectory0.length - 1].reward = reward0;
            }
            for (const step of trajectory1) {
              if (step.reward === 0) step.reward = reward1 * 0.1;
            }
            if (trajectory1.length > 0) {
              trajectory1[trajectory1.length - 1].reward = reward1;
            }
          }

          roundOver = true;
          continue;
        }

        // Discard cards
        const cardsToDiscard = discardDecision.cards || [me.hand[0]];
        // Validate and execute discard
        const validCombos = enumerateValidDiscards(me.hand);
        const isValid = validCombos.some(combo =>
          combo.length === cardsToDiscard.length &&
          combo.every(c => cardsToDiscard.some(d => d.suit === c.suit && d.rank === c.rank))
        );

        const actualDiscard = isValid ? cardsToDiscard : [me.hand[me.hand.length - 1]];

        if (recordTrajectory) {
          const cardsMask = new Array(52).fill(false);
          for (const card of actualDiscard) {
            cardsMask[cardToIndex(card)] = true;
          }
          trajectories[currentPlayer].push({
            state: encodeState(discardSnapshot),
            snapshot: discardSnapshot,
            action: { phase: 'discard', type: 'discard', cards: [...actualDiscard], cardsMask },
            reward: 0,
          });
        }

        // Remove cards from hand, add to discard pile
        for (const card of actualDiscard) {
          const idx = me.hand.findIndex(c => c.suit === card.suit && c.rank === card.rank);
          if (idx !== -1) me.hand.splice(idx, 1);
        }
        discardPile.push(...actualDiscard);

        // --- Draw Phase ---
        const drawHandValue = CardUtils.getHandValue(me.hand);
        const drawTopDiscard = discardPile.length > 0
          ? discardPile[discardPile.length - 1]
          : null;

        const drawSnapshot: GameSnapshot = {
          hand: [...me.hand],
          topDiscard: drawTopDiscard,
          handValue: drawHandValue,
          handSize: me.hand.length,
          opponentHandSize: opp.hand.length,
          myScore: me.score,
          opponentScore: opp.score,
          canCallYaniv: false,
          roundNumber: roundNum,
          discardPileSize: discardPile.length,
          turnPhase: 'draw',
        };

        const drawSource = policies[currentPlayer].decideDraw(drawSnapshot);

        if (recordTrajectory) {
          trajectories[currentPlayer].push({
            state: encodeState(drawSnapshot),
            snapshot: drawSnapshot,
            action: {
              phase: 'draw',
              type: drawSource === 'deck' ? 'draw_deck' : 'draw_discard',
            },
            reward: 0,
          });
        }

        if (drawSource === 'discard' && discardPile.length > 0) {
          const drawn = discardPile.pop()!;
          me.hand.push(drawn);
        } else {
          // Draw from deck
          const drawn = deck.draw(1);
          if (drawn.length > 0) {
            me.hand.push(drawn[0]);
          } else if (discardPile.length > 1) {
            // Reshuffle discard pile into deck
            const top = discardPile.pop()!;
            deck.addCards(discardPile.splice(0));
            discardPile.push(top);
            deck.shuffle();
            const redrawn = deck.draw(1);
            if (redrawn.length > 0) me.hand.push(redrawn[0]);
          }
        }

        currentPlayer = 1 - currentPlayer;
      }

      // Check for game end
      if (players[0].score >= 101 || players[1].score >= 101) {
        break;
      }
    }

    // Determine winner (lower score wins)
    const winner: 0 | 1 = players[0].score <= players[1].score ? 0 : 1;

    return {
      winner,
      scores: [players[0].score, players[1].score],
      rounds: roundNum,
      trajectory0,
      trajectory1,
    };
  }

  /**
   * Play multiple games and return aggregate win count.
   * Alternates who goes first.
   */
  static playMatch(
    policy0: SimPolicy,
    policy1: SimPolicy,
    numGames: number,
  ): { wins0: number; wins1: number } {
    let wins0 = 0;
    let wins1 = 0;

    for (let i = 0; i < numGames; i++) {
      // Alternate who goes first
      const result = i % 2 === 0
        ? GameSimulator.playGame(policy0, policy1)
        : GameSimulator.playGame(policy1, policy0);

      if (i % 2 === 0) {
        if (result.winner === 0) wins0++;
        else wins1++;
      } else {
        if (result.winner === 1) wins0++;
        else wins1++;
      }
    }

    return { wins0, wins1 };
  }
}
