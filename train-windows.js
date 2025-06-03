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

async function windowsTraining() {
  await tf.ready();
  
  console.log('\n========================================');
  console.log('    YANIV AI TRAINING SYSTEM');
  console.log('========================================\n');
  
  const episodes = parseInt(process.argv[2]) || 10;
  
  console.log(`Episodes: ${episodes}`);
  console.log(`Backend: ${tf.getBackend()}`);
  console.log(`\nIMPORTANT: First episode takes 10-30 seconds`);
  console.log(`Each dot (.) = 1 game turn\n`);
  console.log('----------------------------------------');
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  
  let completed = 0;
  let turnCount = 0;
  let dotCount = 0;
  let losses = [];
  let invalidMoves = 0;
  
  // Capture game events and show progress
  const originalLog = console.log;
  const originalWrite = process.stdout.write;
  
  console.log = (...args) => {
    const message = args.join(' ');
    
    // Count turns and show dots
    if (message.includes('Turn')) {
      turnCount++;
      dotCount++;
      if (dotCount % 10 === 0) {
        originalWrite('.');
      }
    }
    
    // Count invalid moves
    if (message.includes('LEARNED: Invalid')) {
      invalidMoves++;
    }
    
    // Show episode completion
    if (message.includes('Episode ended')) {
      // Clear the dots and show completion
      originalWrite('\n');
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      originalLog(`Episode ${completed + 1} complete! (${turnCount} turns, ${elapsed}s total)`);
      turnCount = 0;
      dotCount = 0;
    }
  };
  
  // Show we're starting
  console.log('Episode 1 starting');
  
  try {
    // Training loop
    for (let i = 0; i < episodes; i++) {
      await trainingSystem.runSelfPlayEpisode();
      completed = i + 1;
      
      // Train every 5 episodes for this demo
      if (completed % 5 === 0 || completed === episodes) {
        originalLog(`\nTraining neural network...`);
        const loss = await trainingSystem.train();
        losses.push(loss);
        originalLog(`Loss: ${loss.toFixed(4)}`);
        originalLog(`Invalid moves learned: ${invalidMoves}\n`);
        
        if (completed < episodes) {
          originalLog(`Episode ${completed + 1} starting`);
        }
      } else if (completed < episodes) {
        originalLog(`Episode ${completed + 1} starting`);
      }
    }
    
  } finally {
    console.log = originalLog;
    process.stdout.write = originalWrite;
  }
  
  // Show summary
  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  const avgTime = (totalTime / episodes).toFixed(1);
  
  console.log('\n========================================');
  console.log('    TRAINING COMPLETE!');
  console.log('========================================\n');
  
  console.log(`Total episodes: ${episodes}`);
  console.log(`Total time: ${totalTime} seconds`);
  console.log(`Average time per episode: ${avgTime} seconds`);
  
  if (losses.length > 0) {
    const avgLoss = losses.reduce((a, b) => a + b) / losses.length;
    console.log(`Average training loss: ${avgLoss.toFixed(4)}`);
  }
  
  console.log(`Invalid moves learned from: ${invalidMoves}`);
  
  // Save model
  const modelDir = `./models/yaniv-ai-test-${episodes}`;
  if (!fs.existsSync(modelDir)) {
    fs.mkdirSync(modelDir, { recursive: true });
  }
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, modelDir);
    console.log(`\nModel saved to: ${modelDir}`);
  }
  
  console.log('\n----------------------------------------');
  console.log('Estimate for 4000 episodes:');
  console.log(`Time needed: ${(avgTime * 4000 / 3600).toFixed(1)} hours`);
  console.log('----------------------------------------\n');
  
  process.exit(0);
}

// Show TensorFlow warning then start
console.log('============================');
console.log('Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details.');
console.log('============================');

windowsTraining().catch(error => {
  console.error('\nError:', error);
  process.exit(1);
});