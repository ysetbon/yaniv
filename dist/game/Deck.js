"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Deck = void 0;
const Card_1 = require("./Card");
class Deck {
    constructor() {
        this.cards = [];
        this.initialize();
        this.shuffle();
    }
    initialize() {
        const suits = ['♠', '♥', '♦', '♣'];
        const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
        this.cards = [];
        for (const suit of suits) {
            for (const rank of ranks) {
                this.cards.push(Card_1.CardUtils.create(suit, rank));
            }
        }
    }
    shuffle() {
        for (let i = this.cards.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [this.cards[i], this.cards[j]] = [this.cards[j], this.cards[i]];
        }
    }
    draw(count = 1) {
        const drawn = [];
        for (let i = 0; i < count && this.cards.length > 0; i++) {
            const card = this.cards.pop();
            if (card)
                drawn.push(card);
        }
        return drawn;
    }
    addCards(cards) {
        this.cards.push(...cards);
    }
    getSize() {
        return this.cards.length;
    }
    isEmpty() {
        return this.cards.length === 0;
    }
}
exports.Deck = Deck;
