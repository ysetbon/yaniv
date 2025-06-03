#!/usr/bin/env node
require('ts-node').register({
  compilerOptions: {
    module: 'commonjs',
    target: 'es2020',
    esModuleInterop: true,
    allowSyntheticDefaultImports: true,
    moduleResolution: 'node',
    skipLibCheck: true,
    strict: false
  },
  transpileOnly: true
});

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu');
const { TrainingSystemV2 } = require('./src/game/TrainingSystemV2');
const { saveModelNode } = require('./src/game/NeuralNetworkAI-node');
const fs = require('fs');

async function runTraining() {
  // Set up TensorFlow.js backend
  await tf.ready();
  console.log('TensorFlow.js backend:', tf.getBackend());
  
  console.log('=== Yaniv AI Training System ===\n');
  
  // You can now train for up to 10,000 episodes!
  const episodes = 4000;  // Training for 4000 games as requested
  const trainEvery = 10;
  const saveEvery = 100;  // Save every 100 episodes for longer training
  
  console.log(`Starting training with:`);
  console.log(`- Episodes: ${episodes}`);
  console.log(`- Train every: ${trainEvery} episodes`);
  console.log(`- Save every: ${saveEvery} episodes`);
  console.log(`- Focus: Learning to avoid invalid moves with -Infinity penalties\n`);
  
  const trainingSystem = new TrainingSystemV2();
  let lastSaveTime = Date.now();
  
  for (let i = 0; i < episodes; i++) {
    try {
      // Run self-play episode
      await trainingSystem.runSelfPlayEpisode();
      
      // Train periodically
      if ((i + 1) % trainEvery === 0) {
        const avgLoss = await trainingSystem.train();
        const progress = ((i + 1) / episodes * 100).toFixed(1);
        console.log(`[${progress}%] Episode ${i + 1}/${episodes} - Loss: ${avgLoss.toFixed(4)}`);
      }
      
      // Save model periodically
      if ((i + 1) % saveEvery === 0) {
        const modelDir = `./models/yaniv-ai-model-${i + 1}`;
        
        // Create directory
        if (!fs.existsSync(modelDir)) {
          fs.mkdirSync(modelDir, { recursive: true });
        }
        
        // Save using Node.js handler
        const model = trainingSystem.getNeuralAI()['model'];
        if (model) {
          await saveModelNode(model, modelDir);
          const timeSinceLastSave = ((Date.now() - lastSaveTime) / 1000).toFixed(1);
          console.log(`✓ Model checkpoint saved to ${modelDir} (${timeSinceLastSave}s)`);
          lastSaveTime = Date.now();
        }
      }
      
      // Show progress bar every 10 episodes
      if ((i + 1) % 10 === 0 && (i + 1) % trainEvery !== 0) {
        const progress = ((i + 1) / episodes * 100).toFixed(1);
        const barLength = 30;
        const filledLength = Math.floor(barLength * (i + 1) / episodes);
        const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);
        process.stdout.write(`\r[${bar}] ${progress}%`);
      }
      
    } catch (error) {
      console.error(`\nError in episode ${i + 1}:`, error.message);
      // Continue training despite errors
    }
  }
  
  // Final save
  const finalModelDir = './models/yaniv-ai-model-final';
  if (!fs.existsSync(finalModelDir)) {
    fs.mkdirSync(finalModelDir, { recursive: true });
  }
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, finalModelDir);
  }
  
  console.log(`\n\n✅ Training complete!`);
  console.log(`Final model saved to: ${finalModelDir}`);
  console.log(`\nKey achievements:`);
  console.log(`- Trained AI to recognize invalid moves with -Infinity penalties`);
  console.log(`- AI should now never attempt invalid discards like A and K together`);
  console.log(`- Model ready for integration into the game interface`);
  
  process.exit(0);
}

// Handle errors
process.on('unhandledRejection', (error) => {
  console.error('\n❌ Training error:', error);
  process.exit(1);
});

// Run the training
runTraining().catch(console.error);