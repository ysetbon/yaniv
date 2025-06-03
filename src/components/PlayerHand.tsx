import { Player, Card } from '../types/game';
import { CardComponent } from './Card';
import { CardUtils } from '../game/Card';

interface PlayerHandProps {
  player: Player;
  isCurrentTurn: boolean;
  selectedCards: Card[];
  onCardSelect: (card: Card) => void;
  drawnCards: Card[];
  showCards?: boolean;
}

export function PlayerHand({ player, isCurrentTurn, selectedCards, onCardSelect, drawnCards, showCards = false }: PlayerHandProps) {
  const allCards = [...player.hand, ...drawnCards];
  const sortedCards = CardUtils.sortCards(allCards);
  const handValue = CardUtils.getHandValue(player.hand);

  return (
    <div className={`player-hand ${isCurrentTurn ? 'active' : ''}`}>
      <div className="player-info">
        <h3>{player.name}</h3>
        <span className="hand-value">Hand: {handValue} pts</span>
        {isCurrentTurn && <span className="turn-indicator">Your Turn</span>}
      </div>
      <div className="cards">
        {sortedCards.map((card, index) => {
          const isDrawn = drawnCards.some(c => CardUtils.areEqual(c, card));
          const isSelected = selectedCards.some(c => CardUtils.areEqual(c, card));
          
          return (
            <div key={index} className={`card-wrapper ${isDrawn ? 'newly-drawn' : ''}`}>
              <CardComponent
                card={card}
                selected={isSelected}
                onClick={() => isCurrentTurn && onCardSelect(card)}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}