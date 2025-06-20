import { Card, Rank, Suit } from '../types/game';

export class CardUtils {
  static create(suit: Suit, rank: Rank): Card {
    return {
      id: `${suit}-${rank}`,
      suit,
      rank,
      value: this.getRankValue(rank)
    };
  }

  static getRankValue(rank: Rank): number {
    switch (rank) {
      case 'A': return 1;
      case '2': return 2;
      case '3': return 3;
      case '4': return 4;
      case '5': return 5;
      case '6': return 6;
      case '7': return 7;
      case '8': return 8;
      case '9': return 9;
      case '10': return 10;
      case 'J': return 11;
      case 'Q': return 12;
      case 'K': return 13;
    }
  }

  static getHandValue(cards: Card[]): number {
    return cards.reduce((sum, card) => sum + card.value, 0);
  }

  static areEqual(card1: Card, card2: Card): boolean {
    return card1.rank === card2.rank && card1.suit === card2.suit;
  }

  static isSet(cards: Card[]): boolean {
    if (cards.length < 2) return false;
    const rank = cards[0].rank;
    return cards.every(card => card.rank === rank);
  }

  static isRun(cards: Card[]): boolean {
    if (cards.length < 3) return false;
    
    const suit = cards[0].suit;
    if (!cards.every(card => card.suit === suit)) return false;
    
    const sortedCards = [...cards].sort((a, b) => a.value - b.value);
    
    for (let i = 1; i < sortedCards.length; i++) {
      if (sortedCards[i].value !== sortedCards[i - 1].value + 1) {
        return false;
      }
    }
    
    return true;
  }

  static isValidDiscard(cards: Card[]): boolean {
    if (cards.length === 0) return false;
    if (cards.length === 1) return true;
    return this.isSet(cards) || this.isRun(cards);
  }

  static sortCards(cards: Card[]): Card[] {
    return [...cards].sort((a, b) => {
      if (a.suit !== b.suit) {
        const suitOrder = ['♠', '♥', '♦', '♣'];
        return suitOrder.indexOf(a.suit) - suitOrder.indexOf(b.suit);
      }
      return a.value - b.value;
    });
  }
}