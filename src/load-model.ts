#!/usr/bin/env node
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import * as fs from 'fs';
import * as path from 'path';

async function convertModelForWeb(modelPath: string, outputDir: string) {
  console.log(`Loading model from: ${modelPath}`);
  
  // Load the model
  const model = await tf.loadLayersModel(modelPath);
  
  // Create output directory if it doesn't exist
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Save in web format
  const webPath = `file://${path.resolve(outputDir)}`;
  await model.save(webPath);
  
  console.log(`Model converted and saved to: ${outputDir}`);
  console.log('\nTo use in your web app:');
  console.log('1. Copy the model files to your public directory');
  console.log('2. Load with: await tf.loadLayersModel("/path/to/model.json")');
}

// Usage
const args = process.argv.slice(2);
if (args.length < 2) {
  console.log('Usage: npm run load-model <model-path> <output-dir>');
  console.log('Example: npm run load-model file://./yaniv-ai-model-final ./public/models/yaniv-ai');
  process.exit(1);
}

convertModelForWeb(args[0], args[1]).catch(console.error);