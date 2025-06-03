# Yaniv AI Neural Network Training Guide

This guide explains how to train the neural network AI for the Yaniv card game using the terminal.

## Prerequisites

- Node.js installed
- Project dependencies installed (`npm install`)

## Training the AI

### Basic Training

Run the training script with default settings (1000 episodes):

```bash
npm run train
```

### Interactive Training

The training script will prompt you for:
- Number of training episodes (default: 1000)
- Training frequency (default: every 10 episodes)
- Model save frequency (default: every 50 episodes)

### Training Output

During training, you'll see:
- Progress bar showing completion percentage
- Loss values indicating learning quality (lower is better)
- Model checkpoints saved to disk

Example output:
```
[10.0%] Episode 100/1000 - Loss: 0.4521
✓ Model checkpoint saved to file://./yaniv-ai-model-100 (15.2s)
[████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 20.0%
```

## Using the Trained Model

### In Terminal/Node.js

The trained models are saved as:
- `yaniv-ai-model-{episode}` - Checkpoints during training
- `yaniv-ai-model-final` - Final trained model

### Converting for Web Use

To use the trained model in your web application:

```bash
npm run load-model file://./yaniv-ai-model-final ./public/models/yaniv-ai
```

This converts the model to web format and saves it to the specified directory.

### Loading in Your Game

```typescript
// In your web app
const neuralAI = new NeuralNetworkAI();
await neuralAI.loadModel('/models/yaniv-ai/model.json');
```

## Training Parameters

### Episodes
More episodes = better training but longer time
- Quick test: 100 episodes (~2 minutes)
- Basic training: 1000 episodes (~20 minutes)
- Good training: 5000 episodes (~1-2 hours)
- Best results: 10000+ episodes

### Training Frequency
How often to update the neural network:
- Lower values (5-10): More frequent updates, potentially unstable
- Higher values (20-50): More stable but slower learning

### Architecture Details

The neural network uses:
- **Input**: 234-dimensional state encoding
  - One-hot encoded cards in hand (52)
  - Top 3 discard pile cards (156)
  - Game features (hand value, scores, etc.)
- **Hidden Layers**: 256 → 128 → 64 neurons with ReLU activation
- **Output**: 
  - Action probabilities (55 possible actions)
  - State value estimation

## Tips for Better Training

1. **Start Small**: Test with 100-500 episodes first
2. **Monitor Loss**: Loss should decrease over time
3. **Save Checkpoints**: Use frequent saves to avoid losing progress
4. **Multiple Runs**: Train multiple models and compare performance

## Troubleshooting

### High Loss Values
- Normal at start (>1.0)
- Should decrease to <0.5 after good training
- If stuck high, try adjusting learning rate

### Memory Issues
- Reduce batch size in `TrainingSystemV2.ts`
- Train in shorter sessions with saves

### Slow Training
- Ensure you're using `@tensorflow/tfjs-node` (CPU optimized)
- Consider reducing model size for faster iteration

## Advanced Usage

### Custom Training Loop

```typescript
import { TrainingSystemV2 } from './src/game/TrainingSystemV2';

const trainer = new TrainingSystemV2();
await trainer.runTrainingLoop(
  5000,  // episodes
  20,    // train every
);
```

### Hyperparameter Tuning

Edit `NeuralNetworkAI.ts` to adjust:
- Learning rate: `tf.train.adam(0.001)`
- Network architecture: Layer sizes and dropout rates
- Reward structure in `TrainingSystemV2.ts`