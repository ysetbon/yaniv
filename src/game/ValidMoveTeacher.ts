import { Card } from '../types/game';
import { CardUtils } from './Card';

export class ValidMoveTeacher {
  /**
   * Generate examples of invalid discards to teach the AI what NOT to do
   */
  static generateInvalidDiscards(hand: Card[]): { cards: Card[], reason: string }[] {
    const invalidExamples: { cards: Card[], reason: string }[] = [];
    
    if (hand.length < 2) return invalidExamples;
    
    // Example 1: Random unrelated cards (like A and K)
    const highCard = hand.find(c => c.value >= 10);
    const lowCard = hand.find(c => c.value <= 3);
    if (highCard && lowCard && highCard.suit !== lowCard.suit) {
      invalidExamples.push({
        cards: [highCard, lowCard],
        reason: 'Not a valid set or run (A and K)'
      });
    }
    
    // Example 2: Almost a run but not quite (e.g., 3♥, 5♥ - missing 4)
    const sameSuitCards = this.groupBySuit(hand);
    for (const [suit, cards] of sameSuitCards) {
      if (cards.length >= 2) {
        const sorted = cards.sort((a, b) => a.value - b.value);
        for (let i = 0; i < sorted.length - 1; i++) {
          if (sorted[i + 1].value - sorted[i].value === 2) {
            // Gap of 2 - invalid run
            invalidExamples.push({
              cards: [sorted[i], sorted[i + 1]],
              reason: `Invalid run - missing ${sorted[i].value + 1}${suit}`
            });
          }
        }
      }
    }
    
    // Example 3: Almost a set but different ranks (e.g., 7♥, 8♥)
    for (const [suit, cards] of sameSuitCards) {
      if (cards.length >= 2) {
        const sorted = cards.sort((a, b) => a.value - b.value);
        for (let i = 0; i < sorted.length - 1; i++) {
          if (sorted[i + 1].value - sorted[i].value === 1) {
            // Only 2 consecutive cards - not enough for a run
            invalidExamples.push({
              cards: [sorted[i], sorted[i + 1]],
              reason: 'Need 3+ cards for a run'
            });
          }
        }
      }
    }
    
    // Example 4: Mixed suits with same rank but only 2 cards
    const sameRankCards = this.groupByRank(hand);
    for (const [rank, cards] of sameRankCards) {
      if (cards.length === 2) {
        // Valid as a set, but let's sometimes teach it's the minimum
        if (Math.random() < 0.3) {
          invalidExamples.push({
            cards: [cards[0]],
            reason: 'Single card when a set is available'
          });
        }
      }
    }
    
    return invalidExamples;
  }
  
  /**
   * Validate if a discard is legal and explain why/why not
   */
  static validateDiscard(cards: Card[]): { valid: boolean, reason: string } {
    if (!cards || cards.length === 0) {
      return { valid: false, reason: 'No cards selected' };
    }
    
    if (cards.length === 1) {
      return { valid: true, reason: 'Single card' };
    }
    
    // Check for set
    if (CardUtils.isSet(cards)) {
      return { valid: true, reason: `Valid set of ${cards[0].rank}s` };
    }
    
    // Check for run
    if (CardUtils.isRun(cards)) {
      const sorted = cards.sort((a, b) => a.value - b.value);
      return { 
        valid: true, 
        reason: `Valid run: ${sorted[0].rank}-${sorted[sorted.length-1].rank} of ${cards[0].suit}` 
      };
    }
    
    // Invalid combination
    if (cards.length === 2) {
      const [c1, c2] = cards;
      if (c1.rank === c2.rank) {
        return { valid: true, reason: 'Valid pair' };
      }
      if (c1.suit === c2.suit && Math.abs(c1.value - c2.value) === 1) {
        return { valid: false, reason: 'Need 3+ consecutive cards for a run' };
      }
      return { valid: false, reason: 'Not a valid set or run' };
    }
    
    return { valid: false, reason: 'Invalid combination' };
  }
  
  /**
   * Generate training examples with labels
   */
  static generateTrainingExamples(hand: Card[]): {
    cards: Card[],
    valid: boolean,
    score: number,
    reason: string
  }[] {
    const examples: any[] = [];
    
    // Add all valid discards with positive scores
    const validDiscards = this.getAllValidDiscards(hand);
    for (const discard of validDiscards) {
      const discardValue = CardUtils.getHandValue(discard);
      const remainingValue = CardUtils.getHandValue(
        hand.filter(c => !discard.some(d => CardUtils.areEqual(c, d)))
      );
      
      let score = discardValue * 0.5 - remainingValue * 0.3;
      if (discard.length > 1 && (CardUtils.isSet(discard) || CardUtils.isRun(discard))) {
        score += 10; // Bonus for sets/runs
      }
      
      examples.push({
        cards: discard,
        valid: true,
        score: score,
        reason: this.validateDiscard(discard).reason
      });
    }
    
    // Add some invalid discards with negative infinity scores
    const invalidDiscards = this.generateInvalidDiscards(hand);
    for (const invalid of invalidDiscards) {
      examples.push({
        cards: invalid.cards,
        valid: false,
        score: -Infinity, // NEVER choose these
        reason: invalid.reason
      });
    }
    
    return examples;
  }
  
  private static groupBySuit(cards: Card[]): Map<string, Card[]> {
    const groups = new Map<string, Card[]>();
    for (const card of cards) {
      const group = groups.get(card.suit) || [];
      group.push(card);
      groups.set(card.suit, group);
    }
    return groups;
  }
  
  private static groupByRank(cards: Card[]): Map<string, Card[]> {
    const groups = new Map<string, Card[]>();
    for (const card of cards) {
      const group = groups.get(card.rank) || [];
      group.push(card);
      groups.set(card.rank, group);
    }
    return groups;
  }
  
  private static getAllValidDiscards(hand: Card[]): Card[][] {
    const validDiscards: Card[][] = [];
    
    // Single cards
    hand.forEach(card => validDiscards.push([card]));
    
    // Sets
    const byRank = this.groupByRank(hand);
    byRank.forEach(cards => {
      if (cards.length >= 2) {
        for (let size = 2; size <= cards.length; size++) {
          validDiscards.push(cards.slice(0, size));
        }
      }
    });
    
    // Runs
    const bySuit = this.groupBySuit(hand);
    bySuit.forEach(cards => {
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
}