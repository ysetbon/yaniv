import { Card, Rank, Suit } from '../types/game';
import { CardUtils } from './Card';

export class Deck {
  private cards: Card[] = [];

  constructor() {
    this.initialize();
    this.shuffle();
  }

  private initialize(): void {
    const suits: Suit[] = ['♠', '♥', '♦', '♣'];
    const ranks: Rank[] = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
    
    this.cards = [];
    for (const suit of suits) {
      for (const rank of ranks) {
        this.cards.push(CardUtils.create(suit, rank));
      }
    }
  }

  shuffle(): void {
    for (let i = this.cards.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.cards[i], this.cards[j]] = [this.cards[j], this.cards[i]];
    }
  }

  draw(count: number = 1): Card[] {
    const drawn: Card[] = [];
    for (let i = 0; i < count && this.cards.length > 0; i++) {
      const card = this.cards.pop();
      if (card) drawn.push(card);
    }
    return drawn;
  }

  addCards(cards: Card[]): void {
    this.cards.push(...cards);
  }

  getSize(): number {
    return this.cards.length;
  }

  isEmpty(): boolean {
    return this.cards.length === 0;
  }
}