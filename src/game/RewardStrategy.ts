import { Card } from '../types/game';
import { CardUtils } from './Card';

export class RewardStrategy {
  // Constants for reward shaping
  private static readonly YANIV_BASE_REWARD = 100;
  private static readonly ASSAF_PENALTY = -50;
  private static readonly INVALID_MOVE_PENALTY = -Infinity;
  private static readonly PROGRESS_MULTIPLIER = 0.1;
  
  /**
   * Reward for calling Yaniv
   */
  static calculateYanivReward(
    handValue: number, 
    wasSuccessful: boolean,
    wasAssaf: boolean = false
  ): number {
    if (!wasSuccessful) {
      return this.INVALID_MOVE_PENALTY;
    }
    
    if (wasAssaf) {
      return this.ASSAF_PENALTY;
    }
    
    // Reward inversely proportional to hand value
    // Calling with 1 = 100 points, with 7 = 94 points
    return this.YANIV_BASE_REWARD - handValue + 1;
  }
  
  /**
   * Reward for discarding cards
   */
  static calculateDiscardReward(
    handBefore: Card[],
    cardsDiscarded: Card[],
    handAfter: Card[]
  ): number {
    const valueBefore = CardUtils.getHandValue(handBefore);
    const valueAfter = CardUtils.getHandValue(handAfter);
    const discardValue = CardUtils.getHandValue(cardsDiscarded);
    
    let reward = 0;
    
    // 1. Reward for reducing hand value
    const valueReduction = valueBefore - valueAfter;
    reward += valueReduction * 1.5;
    
    // 2. Bonus for discarding sets/runs (they count as 0)
    if (cardsDiscarded.length > 1) {
      if (CardUtils.isSet(cardsDiscarded) || CardUtils.isRun(cardsDiscarded)) {
        reward += 10; // Big bonus for smart discards
      }
    }
    
    // 3. Progress toward Yaniv
    if (valueAfter <= 7) {
      reward += 20; // Ready to call Yaniv!
    } else if (valueAfter <= 15) {
      reward += 10; // Getting close
    } else if (valueAfter <= 25) {
      reward += 5; // Making progress
    }
    
    // 4. Penalty for keeping high cards
    const highCards = handAfter.filter(c => c.value >= 10).length;
    reward -= highCards * 2;
    
    // 5. Small penalty to encourage finishing quickly
    reward -= 0.5;
    
    return reward;
  }
  
  /**
   * Reward for drawing cards
   */
  static calculateDrawReward(
    handBefore: Card[],
    drawnCard: Card,
    drewFromDiscard: boolean,
    couldHaveDrawnFromDiscard: boolean
  ): number {
    let reward = 0;
    
    // Check if the drawn card creates combinations
    const formsSet = this.checkFormsSet(handBefore, drawnCard);
    const formsRun = this.checkFormsRun(handBefore, drawnCard);
    const cardValue = drawnCard.value;
    
    if (drewFromDiscard) {
      // Chose to draw from discard
      if (formsSet || formsRun) {
        reward += 15; // Excellent strategic move
      } else if (cardValue <= 3) {
        reward += 8; // Good - took low card
      } else if (cardValue <= 5) {
        reward += 3; // OK - took medium card
      } else {
        reward -= 5; // Bad - took high card without purpose
      }
    } else {
      // Drew from deck
      if (couldHaveDrawnFromDiscard) {
        // Check what we missed
        const missedOpportunity = formsSet || formsRun || cardValue <= 3;
        if (missedOpportunity) {
          reward -= 10; // Penalty for missing good card
        } else {
          reward += 2; // Good - avoided bad card
        }
      }
      
      // Small penalty for unknown risk
      reward -= 1;
    }
    
    // Penalty based on new hand value
    const newHandValue = CardUtils.getHandValue([...handBefore, drawnCard]);
    if (newHandValue > 40) {
      reward -= 5; // Hand getting too high
    }
    
    return reward;
  }
  
  /**
   * Calculate end-of-round rewards/penalties
   */
  static calculateRoundEndReward(
    finalHandValue: number,
    position: number, // 1st, 2nd, 3rd, 4th
    totalPlayers: number
  ): number {
    let reward = 0;
    
    // Position-based reward
    if (position === 1) {
      reward += 50; // Winner bonus
    } else {
      // Penalty based on position
      reward -= (position - 1) * 10;
    }
    
    // Penalty for high hand value at round end
    reward -= finalHandValue * 0.5;
    
    return reward;
  }
  
  /**
   * Helper: Check if card forms a set with hand
   */
  private static checkFormsSet(hand: Card[], newCard: Card): boolean {
    const sameRank = hand.filter(c => c.rank === newCard.rank);
    return sameRank.length >= 1; // Will form at least a pair
  }
  
  /**
   * Helper: Check if card forms a run with hand
   */
  private static checkFormsRun(hand: Card[], newCard: Card): boolean {
    const sameSuit = hand.filter(c => c.suit === newCard.suit);
    if (sameSuit.length < 2) return false;
    
    const all = [...sameSuit, newCard].sort((a, b) => a.value - b.value);
    
    // Check for consecutive cards
    for (let i = 0; i <= all.length - 3; i++) {
      if (all[i + 1].value === all[i].value + 1 && 
          all[i + 2].value === all[i].value + 2) {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Calculate cumulative reward for state-action evaluation
   */
  static calculateStateValue(
    handValue: number,
    turnsRemaining: number = 10
  ): number {
    // Estimate future value based on current state
    if (handValue <= 7) {
      return 80; // Can call Yaniv soon
    } else if (handValue <= 15) {
      return 40; // Good position
    } else if (handValue <= 25) {
      return 20; // OK position
    } else {
      return -handValue * 0.5; // Bad position
    }
  }
}