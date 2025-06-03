import { Card } from '../types/game';
import { CardComponent } from './Card';

interface DiscardPileProps {
  opponentCard: Card | null;
  totalCards: number;
  onDrawCard: () => void;
  canDraw: boolean;
  isDrawPhase: boolean;
  isHumanTurn: boolean;
}

export function DiscardPile({ opponentCard, totalCards, onDrawCard, canDraw, isDrawPhase, isHumanTurn }: DiscardPileProps) {
  return (
    <div className="discard-pile">
      <h3>Discard Pile</h3>
      
      {opponentCard ? (
        <div className="discard-display">
          <div 
            className={`discard-card-wrapper ${canDraw ? 'can-draw' : ''}`}
            onClick={() => canDraw && onDrawCard()}
            title={canDraw ? 'Draw this card' : ''}
          >
            <CardComponent card={opponentCard} />
            {canDraw && (
              <div className="draw-indicator">↑</div>
            )}
          </div>
          <div className="opponent-label">Opponent's discard</div>
        </div>
      ) : (
        <div className="no-opponent-card">
          {isDrawPhase ? (
            <div className="empty-message">
              No opponent card to draw
              <small>Draw from deck instead</small>
            </div>
          ) : isHumanTurn ? (
            <div className="empty-message">
              No opponent card available
            </div>
          ) : (
            <div className="empty-message">
              Waiting for opponent...
            </div>
          )}
        </div>
      )}
      
      <div className="pile-info">
        <div className="pile-count">{totalCards} total cards</div>
      </div>
    </div>
  );
}