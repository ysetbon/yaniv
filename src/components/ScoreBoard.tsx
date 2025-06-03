import { Player } from '../types/game';

interface ScoreBoardProps {
  players: Player[];
  currentPlayerId: string;
}

export function ScoreBoard({ players, currentPlayerId }: ScoreBoardProps) {
  return (
    <div className="scoreboard">
      <h2>Scores</h2>
      <div className="scores">
        {players.map(player => (
          <div 
            key={player.id} 
            className={`score-item ${player.id === currentPlayerId ? 'current' : ''}`}
          >
            <span className="player-name">{player.name}</span>
            <span className="player-score">{player.score}</span>
            {player.hasCalledYanivLastRound && (
              <span className="yaniv-penalty" title="Called Yaniv last round">⚠️</span>
            )}
          </div>
        ))}
      </div>
      <div className="score-info">
        <span>First to 101 loses</span>
      </div>
    </div>
  );
}