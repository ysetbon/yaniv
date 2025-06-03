#!/usr/bin/env node
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import { TrainingSystemV2 } from './game/TrainingSystemV2';
import { saveModelNode } from './game/NeuralNetworkAI-node';
import * as readline from 'readline';
import * as fs from 'fs';
import * as path from 'path';

// Set up TensorFlow.js backend
async function setupTF() {
  await tf.ready();
  console.log('TensorFlow.js backend:', tf.getBackend());
}

// Initialize TensorFlow
setupTF();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function question(prompt: string): Promise<string> {
  return new Promise(resolve => {
    rl.question(prompt, resolve);
  });
}

async function main() {
  // Ensure TensorFlow is ready
  await tf.ready();
  
  console.log('=== Yaniv AI Training System ===\n');
  
  // Get training parameters
  const episodesInput = await question('Number of training episodes (default 1000, max 10000): ');
  const episodes = Math.min(10000, parseInt(episodesInput || '1000'));
  const trainEvery = parseInt(await question('Train every N episodes (default 10): ') || '10');
  const saveEvery = parseInt(await question('Save model every N episodes (default 50): ') || '50');
  
  console.log(`\nStarting training with:`);
  console.log(`- Episodes: ${episodes}`);
  console.log(`- Train every: ${trainEvery} episodes`);
  console.log(`- Save every: ${saveEvery} episodes\n`);
  
  rl.close();
  
  const trainingSystem = new TrainingSystemV2();
  let lastSaveTime = Date.now();
  
  // Override the training loop to add progress reporting
  for (let i = 0; i < episodes; i++) {
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
      const barLength = 50;
      const filledLength = Math.floor(barLength * (i + 1) / episodes);
      const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);
      process.stdout.write(`\r[${bar}] ${progress}%`);
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
  console.log(`\nTo use the trained model, copy the model files to your web app.`);
  console.log(`Model files: ${finalModelDir}/model.json and ${finalModelDir}/weights.bin`);
  
  process.exit(0);
}

// Handle errors
process.on('unhandledRejection', (error) => {
  console.error('\n❌ Training error:', error);
  process.exit(1);
});

// Run the training
main().catch(console.error);