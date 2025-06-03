import GameBoardAI from './components/GameBoardAI';
import './App.css';

function App() {
  console.log('App-AI rendering');
  
  try {
    return (
      <div className="app">
        <GameBoardAI />
      </div>
    );
  } catch (error) {
    console.error('Error rendering App:', error);
    return (
      <div style={{ padding: '20px', color: 'red' }}>
        <h1>Error</h1>
        <p>{error instanceof Error ? error.message : 'Unknown error'}</p>
      </div>
    );
  }
}

export default App;