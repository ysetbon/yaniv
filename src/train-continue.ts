#!/usr/bin/env node
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import { TrainingSystemV2 } from './game/TrainingSystemV2';
import * as fs from 'fs';

async function continueTraining() {
  await tf.ready();
  console.log('=== Continue Training from Checkpoint ===\n');
  
  const trainingSystem = new TrainingSystemV2();
  
  // Try to load existing model
  const modelPath = process.argv[2] || './models/yaniv-ai-model-1020';
  console.log(`Loading model from: ${modelPath}`);
  
  try {
    const modelJsonPath = `file://${modelPath}/model.json`;
    await trainingSystem.getNeuralAI().loadModel(modelJsonPath);
    console.log('Model loaded successfully!');
  } catch (error) {
    console.log('Could not load model, starting fresh');
  }
  
  // Train for fewer episodes
  const additionalEpisodes = parseInt(process.argv[3] || '200');
  console.log(`Training for ${additionalEpisodes} more episodes...\n`);
  
  await trainingSystem.runTrainingLoop(additionalEpisodes, 10);
  
  // Save final model
  const finalDir = './models/yaniv-ai-model-continued';
  if (!fs.existsSync(finalDir)) {
    fs.mkdirSync(finalDir, { recursive: true });
  }
  
  console.log('\nTraining complete!');
  process.exit(0);
}

continueTraining().catch(console.error);