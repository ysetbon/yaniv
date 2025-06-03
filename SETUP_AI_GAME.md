# Setting Up the AI Game

## 1. Copy Your Trained Model

After training is complete, copy your model files to the public directory:

```bash
# Create the models directory in public
mkdir -p public/models/yaniv-ai-model-final

# Copy the trained model files
cp models/yaniv-ai-model-final/model.json public/models/yaniv-ai-model-final/
cp models/yaniv-ai-model-final/weights.bin public/models/yaniv-ai-model-final/
```

## 2. Update Your App.tsx

Replace the game initialization in your App.tsx to use the AI-enabled game board:

```typescript
import GameBoardAI from './components/GameBoardAI';

function App() {
  return (
    <div className="App">
      <h1>Yaniv Card Game - AI Opponent</h1>
      <GameBoardAI />
    </div>
  );
}
```

## 3. Required Updates to NeuralNetworkAI.ts

Update the loadModel method to work in the browser:

```typescript
async loadModel(path: string) {
  try {
    this.model = await tf.loadLayersModel(path);
    console.log('Model loaded successfully from:', path);
  } catch (error) {
    console.error('Failed to load model:', error);
    throw error;
  }
}
```

## 4. Start the Game

```bash
npm run dev
```

## Playing Against the AI

1. **Select AI Mode**: 
   - Rule-based: Original programmed strategy
   - Neural Network: Uses your trained model
   - Hybrid: Combines both (recommended)

2. **Gameplay**:
   - You play first as "You"
   - AI opponent plays automatically
   - Select cards and click "Discard" when it's your turn
   - Draw from deck or discard pile after discarding
   - Call Yaniv when your hand value is 7 or less

3. **AI Behavior**:
   - With trained model: Makes decisions based on learned patterns
   - Without model: Falls back to rule-based strategy
   - Hybrid mode: Uses neural network for complex decisions, rules for safety

## Troubleshooting

### Model Not Loading
- Check browser console for errors
- Ensure model files are in `public/models/yaniv-ai-model-final/`
- Verify both `model.json` and `weights.bin` are present

### AI Not Making Moves
- Check console for JavaScript errors
- Ensure EnhancedComputerAI is properly initialized
- Verify game state transitions are working

### Performance Issues
- Neural network inference can be slow in browser
- Consider using hybrid mode for better performance
- Reduce model complexity if needed