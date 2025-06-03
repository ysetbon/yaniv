import { Card, GameState, Player, GameAction } from '../types/game';
import { CardUtils } from './Card';
import { Deck } from './Deck';

export class YanivGame {
  private state: GameState;
  private deck: Deck;

  constructor(playerNames: string[]) {
    this.deck = new Deck();
    this.state = this.initializeGame(playerNames);
  }

  private initializeGame(playerNames: string[]): GameState {
    const players: Player[] = playerNames.map((name, index) => ({
      id: `player-${index}`,
      name,
      hand: [],
      score: 0,
      hasCalledYanivLastRound: false
    }));

    // Deal 5 cards to each player
    for (const player of players) {
      player.hand = this.deck.draw(5);
    }

    const discardPile = this.deck.draw(1);
    
    // Initialize opponent discards map
    const lastOpponentDiscards = new Map<string, Card[]>();
    
    // For the first player's first turn, treat the initial card as if player 2 discarded it
    // This allows player 1 to optionally draw it
    if (discardPile.length > 0) {
      lastOpponentDiscards.set(players[1].id, [...discardPile]);
    }

    return {
      players,
      currentPlayerIndex: 0,
      deck: [],
      discardPile,
      gamePhase: 'playing',
      roundNumber: 1,
      turnPhase: 'discard',
      lastDiscardPlayerId: players[1].id,  // Pretend player 2 discarded it
      lastOpponentDiscards
    };
  }

  getState(): GameState {
    // Return a copy with Map converted to object for serialization
    return { 
      ...this.state,
      lastOpponentDiscards: new Map(this.state.lastOpponentDiscards)
    };
  }

  getCurrentPlayer(): Player {
    return this.state.players[this.state.currentPlayerIndex];
  }

  getLastOpponentDiscard(playerId: string): Card[] | null {
    // Get the opponent's ID
    const opponentId = this.state.players.find(p => p.id !== playerId)?.id;
    if (!opponentId) return null;
    
    // Return the opponent's last discard
    return this.state.lastOpponentDiscards.get(opponentId) || null;
  }

  canCallYaniv(playerId: string): boolean {
    const player = this.state.players.find(p => p.id === playerId);
    if (!player) return false;
    
    const handValue = CardUtils.getHandValue(player.hand);
    return handValue <= 7 && !player.hasCalledYanivLastRound;
  }

  // First phase of turn: discard cards
  discard(playerId: string, cards: Card[]): boolean {
    if (this.getCurrentPlayer().id !== playerId) {
      throw new Error('Not your turn');
    }

    if (this.state.turnPhase !== 'discard') {
      throw new Error('Must be in discard phase');
    }

    const player = this.state.players.find(p => p.id === playerId);
    if (!player) return false;

    // Validate that player has these cards
    for (const card of cards) {
      const index = player.hand.findIndex(c => CardUtils.areEqual(c, card));
      if (index === -1) return false;
    }

    // Validate discard is legal
    if (!CardUtils.isValidDiscard(cards)) {
      return false;
    }

    // Remove cards from hand
    for (const card of cards) {
      const index = player.hand.findIndex(c => CardUtils.areEqual(c, card));
      player.hand.splice(index, 1);
    }

    // Add to discard pile
    this.state.discardPile.push(...cards);

    // Track this player's discard
    this.state.lastDiscardPlayerId = playerId;
    this.state.lastOpponentDiscards.set(playerId, [...cards]);

    // Move to draw phase
    this.state.turnPhase = 'draw';

    return true;
  }

  // Second phase of turn: draw ONE card
  drawFromDeck(playerId: string): Card[] {
    if (this.getCurrentPlayer().id !== playerId) {
      throw new Error('Not your turn');
    }

    if (this.state.turnPhase !== 'draw') {
      throw new Error('Must discard first');
    }

    const cards = this.deck.draw(1);
    if (cards.length === 0 && this.state.discardPile.length > 1) {
      // Reshuffle discard pile into deck
      const topCard = this.state.discardPile.pop()!;
      this.deck.addCards(this.state.discardPile);
      this.deck.shuffle();
      this.state.discardPile = [topCard];
      return this.deck.draw(1);
    }

    const player = this.state.players.find(p => p.id === playerId)!;
    player.hand.push(...cards);

    // End turn
    this.state.turnPhase = 'discard';
    this.nextTurn();

    return cards;
  }

  drawFromDiscard(playerId: string): Card[] {
    if (this.getCurrentPlayer().id !== playerId) {
      throw new Error('Not your turn');
    }

    if (this.state.turnPhase !== 'draw') {
      throw new Error('Must discard first');
    }

    // Get opponent's last discard
    const opponentDiscards = this.getLastOpponentDiscard(playerId);
    if (!opponentDiscards || opponentDiscards.length === 0) {
      throw new Error('No opponent card to draw');
    }

    // Take only the first card from opponent's last discard
    const cardToTake = opponentDiscards[0];
    
    // Find and remove this card from the discard pile
    const cardIndex = this.state.discardPile.findIndex(c => CardUtils.areEqual(c, cardToTake));
    if (cardIndex === -1) {
      throw new Error('Opponent card not found in discard pile');
    }
    
    this.state.discardPile.splice(cardIndex, 1);
    
    const player = this.state.players.find(p => p.id === playerId)!;
    player.hand.push(cardToTake);

    // Clear the opponent's last discard since it was taken
    const opponentId = this.state.players.find(p => p.id !== playerId)?.id;
    if (opponentId) {
      this.state.lastOpponentDiscards.delete(opponentId);
    }

    // End turn
    this.state.turnPhase = 'discard';
    this.nextTurn();

    return [cardToTake];
  }

  callYaniv(playerId: string): { success: boolean; assaf?: boolean; scores?: Record<string, number> } {
    if (!this.canCallYaniv(playerId)) {
      return { success: false };
    }

    const callerIndex = this.state.players.findIndex(p => p.id === playerId);
    const caller = this.state.players[callerIndex];
    const callerValue = CardUtils.getHandValue(caller.hand);

    let lowestValue = callerValue;
    let assaf = false;

    // Check for Assaf
    for (let i = 0; i < this.state.players.length; i++) {
      if (i !== callerIndex) {
        const playerValue = CardUtils.getHandValue(this.state.players[i].hand);
        if (playerValue <= callerValue) {
          assaf = true;
          lowestValue = Math.min(lowestValue, playerValue);
        }
      }
    }

    const scores: Record<string, number> = {};

    // Calculate scores
    for (const player of this.state.players) {
      const handValue = CardUtils.getHandValue(player.hand);
      
      if (player.id === playerId && assaf) {
        scores[player.id] = 30;
        player.hasCalledYanivLastRound = true;
      } else if (handValue === lowestValue) {
        scores[player.id] = 0;
        player.hasCalledYanivLastRound = false;
      } else {
        scores[player.id] = handValue;
        player.hasCalledYanivLastRound = false;
      }

      player.score += scores[player.id];

      // 50-point reduction rule
      if (player.score % 50 === 0 && player.score > 0) {
        player.score -= 50;
      }
    }

    this.checkGameEnd();

    return { success: true, assaf, scores };
  }

  private nextTurn(): void {
    this.state.currentPlayerIndex = (this.state.currentPlayerIndex + 1) % this.state.players.length;
  }

  private checkGameEnd(): void {
    const loser = this.state.players.find(p => p.score >= 101);
    if (loser) {
      this.state.gamePhase = 'gameEnd';
      const winner = this.state.players.find(p => p.id !== loser.id);
      if (winner) {
        this.state.winner = winner.id;
      }
    } else {
      this.startNewRound();
    }
  }

  private startNewRound(): void {
    this.deck = new Deck();
    this.state.roundNumber++;
    this.state.discardPile = this.deck.draw(1);
    this.state.turnPhase = 'discard';
    this.state.lastOpponentDiscards = new Map(); // Clear discards for new round
    
    // Determine who starts this round (alternate or use some rule)
    // For now, let's keep the same starting player
    const startingPlayerIndex = 0;
    this.state.currentPlayerIndex = startingPlayerIndex;
    
    // Treat initial card as if the other player discarded it
    const otherPlayerIndex = startingPlayerIndex === 0 ? 1 : 0;
    this.state.lastDiscardPlayerId = this.state.players[otherPlayerIndex].id;
    
    if (this.state.discardPile.length > 0) {
      this.state.lastOpponentDiscards.set(this.state.players[otherPlayerIndex].id, [...this.state.discardPile]);
    }

    // Deal 5 cards to each player
    for (const player of this.state.players) {
      player.hand = this.deck.draw(5);
    }

    this.state.gamePhase = 'playing';
  }

  makeMove(action: GameAction): boolean {
    if (this.state.gamePhase !== 'playing') return false;

    switch (action.type) {
      case 'yaniv':
        const result = this.callYaniv(action.playerId);
        return result.success;
      default:
        return false;
    }
  }
}