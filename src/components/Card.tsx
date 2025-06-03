import { Card } from '../types/game';

interface CardComponentProps {
  card: Card;
  selected?: boolean;
  onClick?: () => void;
  faceDown?: boolean;
}

export function CardComponent({ card, selected = false, onClick, faceDown = false }: CardComponentProps) {
  if (faceDown) {
    return <div className="card card-back" onClick={onClick}></div>;
  }

  const isRed = card.suit === '♥' || card.suit === '♦';
  
  return (
    <div 
      className={`card ${isRed ? 'red' : 'black'} ${selected ? 'selected' : ''}`}
      onClick={onClick}
    >
      <div className="card-corner top-left">
        <div className="card-rank">{card.rank}</div>
        <div className="card-suit">{card.suit}</div>
      </div>
      <div className="card-center">
        <div className="card-suit-large">{card.suit}</div>
      </div>
      <div className="card-corner bottom-right">
        <div className="card-rank">{card.rank}</div>
        <div className="card-suit">{card.suit}</div>
      </div>
    </div>
  );
}