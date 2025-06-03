import { Player } from '../types/game';
import { CardUtils } from '../game/Card';

interface GameControlsProps {
  currentPlayer: Player;
  canDiscard: boolean;
  canCallYaniv: boolean;
  onDiscard: () => void;
  onCallYaniv: () => void;
  turnPhase?: 'discard' | 'draw';
}

export function GameControls({ 
  currentPlayer, 
  canDiscard, 
  canCallYaniv, 
  onDiscard, 
  onCallYaniv,
  turnPhase
}: GameControlsProps) {
  const handValue = CardUtils.getHandValue(currentPlayer.hand);

  return (
    <div className="game-controls">
      {turnPhase === 'discard' && (
        <>
          <button 
            className="discard-button" 
            onClick={onDiscard}
            disabled={!canDiscard}
          >
            Discard Selected
          </button>
          
          <button 
            className="yaniv-button" 
            onClick={onCallYaniv}
            disabled={!canCallYaniv}
            title={handValue > 7 ? `Hand value too high (${handValue})` : 'Call Yaniv!'}
          >
            Call Yaniv! ({handValue} pts)
          </button>
        </>
      )}
      
      {turnPhase === 'draw' && (
        <div className="draw-phase-hint">
          <div>Draw Phase</div>
          <small>Click deck or opponent's discard</small>
        </div>
      )}
    </div>
  );
}