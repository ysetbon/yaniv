import { Card } from '../types/game';
import { CardUtils } from './Card';

export class MoveAnalyzer {
  static analyzeDiscard(
    hand: Card[], 
    discarded: Card[], 
    remainingHand: Card[]
  ): string {
    const handValueBefore = CardUtils.getHandValue(hand);
    const handValueAfter = CardUtils.getHandValue(remainingHand);
    const discardValue = CardUtils.getHandValue(discarded);
    const reduction = handValueBefore - handValueAfter;
    
    let analysis = '';
    
    // Check if it was a set or run
    if (discarded.length > 1) {
      if (CardUtils.isSet(discarded)) {
        analysis += '✓ Good: Discarded a set (worth 0 in hand). ';
      } else if (CardUtils.isRun(discarded)) {
        analysis += '✓ Good: Discarded a run (worth 0 in hand). ';
      }
    }
    
    // Check value reduction
    if (reduction > 10) {
      analysis += `✓ Excellent: Reduced hand by ${reduction} points. `;
    } else if (reduction > 5) {
      analysis += `✓ Good: Reduced hand by ${reduction} points. `;
    } else if (reduction > 0) {
      analysis += `→ OK: Reduced hand by ${reduction} points. `;
    } else {
      analysis += `✗ Poor: No hand value reduction. `;
    }
    
    // Check if keeping low cards
    const highCardsInHand = remainingHand.filter(c => c.value >= 10).length;
    if (highCardsInHand > 2) {
      analysis += `⚠ Warning: Still holding ${highCardsInHand} high cards. `;
    }
    
    // Check for Yaniv readiness
    if (handValueAfter <= 7) {
      analysis += '★ Ready to call Yaniv! ';
    } else if (handValueAfter <= 15) {
      analysis += `→ Getting close (${handValueAfter} points). `;
    }
    
    return analysis || 'Move analyzed.';
  }
  
  static analyzeDrawDecision(
    visibleCard: Card | null,
    drewFromDiscard: boolean,
    hand: Card[]
  ): string {
    if (!visibleCard) {
      return drewFromDiscard ? '✗ Error: No visible card to draw' : '→ Drew from deck (no discard available)';
    }
    
    let analysis = '';
    const cardValue = visibleCard.value;
    
    // Check if card would form combinations
    const wouldFormSet = hand.filter(c => c.rank === visibleCard.rank).length >= 1;
    const sameSuitCards = hand.filter(c => c.suit === visibleCard.suit);
    const wouldFormRun = this.checkPotentialRun(sameSuitCards, visibleCard);
    
    if (drewFromDiscard) {
      if (wouldFormSet || wouldFormRun) {
        analysis += '✓ Excellent: Drew card that forms combination. ';
      } else if (cardValue <= 3) {
        analysis += '✓ Good: Drew low value card. ';
      } else if (cardValue <= 5) {
        analysis += '→ OK: Drew medium value card. ';
      } else {
        analysis += '✗ Questionable: Drew high value card without forming combination. ';
      }
    } else {
      // Didn't draw from discard
      if (wouldFormSet || wouldFormRun) {
        analysis += '✗ Missed opportunity: Card would form combination! ';
      } else if (cardValue <= 3) {
        analysis += '⚠ Missed low value card. ';
      } else if (cardValue >= 8) {
        analysis += '✓ Good: Avoided high value card. ';
      } else {
        analysis += '→ Reasonable: Avoided medium value card. ';
      }
    }
    
    return analysis;
  }
  
  private static checkPotentialRun(sameSuitCards: Card[], newCard: Card): boolean {
    if (sameSuitCards.length < 2) return false;
    
    const allCards = [...sameSuitCards, newCard].sort((a, b) => a.value - b.value);
    
    for (let i = 0; i < allCards.length - 2; i++) {
      if (allCards[i + 1].value === allCards[i].value + 1 &&
          allCards[i + 2].value === allCards[i].value + 2) {
        return true;
      }
    }
    
    return false;
  }
}