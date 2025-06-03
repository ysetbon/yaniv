import React from 'react';
import { Card } from '../types/game';
import './GameLog.css';

export interface LogEntry {
  player: string;
  action: string;
  cards?: Card[];
  timestamp: string;
  details?: string;
}

interface GameLogProps {
  entries: LogEntry[];
}

export function GameLog({ entries }: GameLogProps) {
  const formatCard = (card: Card) => `${card.rank}${card.suit}`;
  
  const formatCards = (cards: Card[]) => {
    return cards.map(formatCard).join(', ');
  };
  
  return (
    <div className="game-log">
      <h3>Game Log</h3>
      <div className="log-entries">
        {entries.slice(-10).reverse().map((entry, index) => (
          <div key={index} className={`log-entry ${entry.player === 'You' ? 'human' : 'ai'}`}>
            <span className="timestamp">{entry.timestamp}</span>
            <span className="player">{entry.player}:</span>
            <span className="action">{entry.action}</span>
            {entry.cards && (
              <span className="cards">[{formatCards(entry.cards)}]</span>
            )}
            {entry.details && (
              <span className="details">({entry.details})</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}