import { Card } from '../types/game';
import { CardUtils } from './Card';

export class ComputerAI {
  static makeDecision(
    hand: Card[], 
    discardPile: Card[],
    canCallYaniv: boolean,
    turnPhase: 'draw' | 'discard' = 'discard',
    opponentHandSize?: number
  ): {
    action: 'yaniv' | 'draw' | 'discard';
    drawSource?: 'deck' | 'discard';
    drawCount?: number;
    cardsToDiscard?: Card[];
  } {
    // Validate based on turn phase
    if (turnPhase === 'draw') {
      // Can only draw during draw phase
      return this.makeDrawDecision(discardPile);
    }
    const handValue = CardUtils.getHandValue(hand);
    
    // Check if calling Yaniv is risky based on opponent's hand size
    const isRiskyYaniv = opponentHandSize !== undefined && opponentHandSize <= 2 && handValue > 5;
    
    // Always call Yaniv if possible and hand value is very low (unless risky)
    if (canCallYaniv && handValue <= 5 && !isRiskyYaniv) {
      return { action: 'yaniv' };
    }

    // Never call Yaniv with 7 if opponent has 1-2 cards
    // Only call with 6-7 if opponent has more cards
    if (canCallYaniv && handValue <= 7 && !isRiskyYaniv) {
      const yanivProbability = opponentHandSize === undefined || opponentHandSize > 3 ? 0.3 : 0.1;
      if (Math.random() < yanivProbability) {
        return { action: 'yaniv' };
      }
    }

    // Decide draw source
    const topDiscards = discardPile.slice(-3).reverse();
    let drawDecision: { source: 'deck' | 'discard'; count?: number } = { source: 'deck' };

    // Check if discard pile has useful cards
    if (topDiscards.length > 0) {
      // Look for cards that could form sets or runs
      for (let i = 0; i < topDiscards.length; i++) {
        const potentialCards = topDiscards.slice(0, i + 1);
        const potentialHand = [...hand, ...potentialCards];
        
        if (this.hasGoodCombinations(potentialHand)) {
          drawDecision = { source: 'discard', count: i + 1 };
          break;
        }
      }
    }

    return {
      action: 'draw',
      drawSource: drawDecision.source,
      drawCount: drawDecision.count
    };
  }

  static selectDiscard(hand: Card[], requiredValue: number): Card[] | null {
    const sortedHand = CardUtils.sortCards(hand);
    
    // Try single cards
    for (const card of sortedHand) {
      if (card.value === requiredValue) {
        return [card];
      }
    }

    // Try sets
    const sets = this.findSets(sortedHand);
    for (const set of sets) {
      if (CardUtils.getHandValue(set) === requiredValue) {
        return set;
      }
    }

    // Try runs
    const runs = this.findRuns(sortedHand);
    for (const run of runs) {
      if (CardUtils.getHandValue(run) === requiredValue) {
        return run;
      }
    }

    // Try combinations of valid discards
    const allValidDiscards = [
      ...sortedHand.map(c => [c]),
      ...sets,
      ...runs
    ];

    for (let i = 0; i < allValidDiscards.length; i++) {
      for (let j = i + 1; j < allValidDiscards.length; j++) {
        const combined = [...allValidDiscards[i], ...allValidDiscards[j]];
        if (CardUtils.getHandValue(combined) === requiredValue && 
            this.areCardsAvailable(combined, hand)) {
          return combined;
        }
      }
    }

    return null;
  }

  private static hasGoodCombinations(cards: Card[]): boolean {
    const sets = this.findSets(cards);
    const runs = this.findRuns(cards);
    
    // Check if new cards help form better combinations
    return sets.length > 0 || runs.length > 0;
  }

  private static findSets(cards: Card[]): Card[][] {
    const sets: Card[][] = [];
    const cardsByRank = new Map<string, Card[]>();

    for (const card of cards) {
      const existing = cardsByRank.get(card.rank) || [];
      existing.push(card);
      cardsByRank.set(card.rank, existing);
    }

    for (const [_, cardsOfRank] of cardsByRank) {
      if (cardsOfRank.length >= 2) {
        // Add all possible combinations
        for (let size = 2; size <= cardsOfRank.length; size++) {
          sets.push(cardsOfRank.slice(0, size));
        }
      }
    }

    return sets;
  }

  private static findRuns(cards: Card[]): Card[][] {
    const runs: Card[][] = [];
    const cardsBySuit = new Map<string, Card[]>();

    for (const card of cards) {
      const existing = cardsBySuit.get(card.suit) || [];
      existing.push(card);
      cardsBySuit.set(card.suit, existing);
    }

    for (const [_, cardsOfSuit] of cardsBySuit) {
      const sorted = cardsOfSuit.sort((a, b) => a.value - b.value);
      
      for (let start = 0; start < sorted.length - 2; start++) {
        for (let end = start + 2; end < sorted.length; end++) {
          const potentialRun = sorted.slice(start, end + 1);
          if (CardUtils.isRun(potentialRun)) {
            runs.push(potentialRun);
          }
        }
      }
    }

    return runs;
  }
  
  static makeStrategicDrawDecision(
    hand: Card[],
    visibleCard: Card | null
  ): {
    action: 'draw';
    drawSource: 'deck' | 'discard';
    drawCount?: number;
    reasoning?: string;
  } {
    if (!visibleCard) {
      return { action: 'draw', drawSource: 'deck' };
    }

    // Check if the visible card would form a set with existing cards
    const cardsWithSameRank = hand.filter(c => c.rank === visibleCard.rank);
    if (cardsWithSameRank.length >= 1) {
      // We can form a set of 2+ cards!
      return { 
        action: 'draw', 
        drawSource: 'discard', 
        drawCount: 1,
        reasoning: `Forms set with ${cardsWithSameRank.length} ${visibleCard.rank}(s)`
      };
    }

    // Check if the visible card would help form a run
    const sameSuitCards = hand.filter(c => c.suit === visibleCard.suit);
    const allSameSuit = [...sameSuitCards, visibleCard].sort((a, b) => a.value - b.value);
    
    // Check for potential runs
    for (let i = 0; i < allSameSuit.length - 2; i++) {
      let runLength = 1;
      for (let j = i + 1; j < allSameSuit.length; j++) {
        if (allSameSuit[j].value === allSameSuit[j-1].value + 1) {
          runLength++;
        } else {
          break;
        }
      }
      if (runLength >= 3) {
        return { 
          action: 'draw', 
          drawSource: 'discard', 
          drawCount: 1,
          reasoning: `Forms run of ${runLength} cards in ${visibleCard.suit}`
        };
      }
    }

    // Check if it's close to forming a run (within 1 card)
    for (const card of sameSuitCards) {
      const diff = Math.abs(card.value - visibleCard.value);
      if (diff === 2) {
        // There's a gap of 1 card - might complete a run later
        return { 
          action: 'draw', 
          drawSource: 'discard', 
          drawCount: 1,
          reasoning: `Close to run with ${card.rank}${card.suit}`
        };
      }
    }

    // Consider low value cards
    if (visibleCard.value <= 3) {
      return { 
        action: 'draw', 
        drawSource: 'discard', 
        drawCount: 1,
        reasoning: `Low value card (${visibleCard.value})`
      };
    }

    // Consider medium value cards if they're better than our highest card
    const highestCard = hand.reduce((max, card) => card.value > max.value ? card : max, hand[0]);
    if (visibleCard.value <= 5 && visibleCard.value < highestCard.value) {
      return { 
        action: 'draw', 
        drawSource: 'discard', 
        drawCount: 1,
        reasoning: `Better than highest card (${highestCard.rank})`
      };
    }

    // Default: draw from deck
    return { action: 'draw', drawSource: 'deck', reasoning: 'No strategic advantage' };
  }
  
  private static makeDrawDecision(discardPile: Card[]): {
    action: 'draw';
    drawSource: 'deck' | 'discard';
    drawCount?: number;
  } {
    // Simple logic for draw decision (fallback)
    const topDiscards = discardPile.slice(-1);
    
    if (topDiscards.length > 0) {
      const topCard = topDiscards[0];
      // Take low value cards or sometimes medium cards
      if (topCard.value <= 3 || (topCard.value <= 5 && Math.random() < 0.3)) {
        return { action: 'draw', drawSource: 'discard', drawCount: 1 };
      }
    }
    
    return { action: 'draw', drawSource: 'deck' };
  }

  private static areCardsAvailable(needed: Card[], available: Card[]): boolean {
    const availableCopy = [...available];
    
    for (const card of needed) {
      const index = availableCopy.findIndex(c => CardUtils.areEqual(c, card));
      if (index === -1) return false;
      availableCopy.splice(index, 1);
    }
    
    return true;
  }
}