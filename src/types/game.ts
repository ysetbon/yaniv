export type Suit = '♠' | '♥' | '♦' | '♣';
export type Rank = 'A' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K';

export interface Card {
  id?: string;
  suit: Suit;
  rank: Rank;
  value: number;
}

export interface Player {
  id: string;
  name: string;
  hand: Card[];
  score: number;
  hasCalledYanivLastRound: boolean;
}

export interface GameState {
  players: Player[];
  currentPlayerIndex: number;
  deck: Card[];
  discardPile: Card[];
  gamePhase: 'waiting' | 'playing' | 'roundEnd' | 'gameEnd';
  roundNumber: number;
  lastAction?: GameAction;
  winner?: string;
  turnPhase?: 'discard' | 'draw';
  lastDiscardPlayerId?: string;
  lastOpponentDiscards: Map<string, Card[]>; // Track last discard for each player
  scores: Record<string, number>;
  targetScore: number;
  roundWinner?: string;
}

export interface GameAction {
  type: 'draw' | 'discard' | 'yaniv';
  playerId: string;
  cards?: Card[];
  source?: 'deck' | 'discard';
}

export interface DrawAction {
  source: 'deck' | 'discard';
  cards: Card[];
}

export interface DiscardAction {
  cards: Card[];
  isValid: boolean;
}

export type AIType = 'rule-based' | 'neural-network' | 'python-trained' | 'hybrid' | 'enhanced-neural';

export interface AIPlayer {
  type: AIType;
  makeMove(gameState: GameState): Promise<any>;
}