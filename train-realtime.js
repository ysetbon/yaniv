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
const readline = require('readline');

async function realtimeTraining() {
  await tf.ready();
  
  console.log('\n🎮 YANIV AI TRAINING - REAL-TIME PROGRESS\n');
  
  const episodes = parseInt(process.argv[2]) || 4000;
  
  console.log(`Episodes to train: ${episodes}`);
  console.log(`Backend: ${tf.getBackend()}`);
  console.log('\nNOTE: Each episode is a complete game and takes 5-30 seconds on CPU\n');
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  
  let completed = 0;
  let currentTurn = 0;
  let gameEvents = [];
  let losses = [];
  
  // Capture game events
  const originalLog = console.log;
  console.log = (...args) => {
    const message = args.join(' ');
    if (message.includes('Turn')) {
      currentTurn = parseInt(message.match(/Turn (\d+)/)?.[1] || 0);
    }
    if (message.includes('Episode ended') || 
        message.includes('LEARNED') || 
        message.includes('calls Yaniv')) {
      gameEvents.push(message.substring(0, 50) + '...');
      if (gameEvents.length > 5) gameEvents.shift();
    }
  };
  
  // Progress update function
  function updateProgress() {
    readline.cursorTo(process.stdout, 0, 6); // Move to line 6
    readline.clearScreenDown(process.stdout);
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
    const progress = completed / episodes;
    const percentage = (progress * 100).toFixed(1);
    
    // Progress bar
    const barLength = 40;
    const filled = Math.floor(barLength * progress);
    const bar = '█'.repeat(filled) + '░'.repeat(barLength - filled);
    
    console.log(`Progress: [${bar}] ${percentage}%`);
    console.log(`Episodes: ${completed}/${episodes} | Time: ${elapsed}s`);
    
    if (completed > 0) {
      const avgTime = elapsed / completed;
      const eta = ((episodes - completed) * avgTime).toFixed(0);
      console.log(`Speed: ${avgTime.toFixed(1)}s/episode | ETA: ${eta}s`);
    } else {
      console.log(`Currently: Episode 1, Turn ${currentTurn}...`);
    }
    
    if (losses.length > 0) {
      const avgLoss = losses.reduce((a, b) => a + b) / losses.length;
      console.log(`\nTraining Loss: ${avgLoss.toFixed(4)}`);
    }
    
    if (gameEvents.length > 0) {
      console.log('\nRecent Events:');
      gameEvents.forEach(event => console.log(`  • ${event}`));
    }
  }
  
  // Update display every 250ms
  const progressInterval = setInterval(updateProgress, 250);
  
  try {
    // Initial display
    console.log('Starting training...\n');
    updateProgress();
    
    // Training loop
    for (let i = 0; i < episodes; i++) {
      currentTurn = 0;
      await trainingSystem.runSelfPlayEpisode();
      completed = i + 1;
      
      // Train every 40 episodes
      if (completed % 40 === 0) {
        const loss = await trainingSystem.train();
        losses.push(loss);
      }
      
      // Save every 100 episodes
      if (completed % 100 === 0) {
        clearInterval(progressInterval);
        readline.cursorTo(process.stdout, 0, 20);
        originalLog(`\n💾 Saving checkpoint at episode ${completed}...`);
        
        const modelDir = `./models/yaniv-ai-model-${completed}`;
        if (!fs.existsSync(modelDir)) {
          fs.mkdirSync(modelDir, { recursive: true });
        }
        
        const model = trainingSystem.getNeuralAI()['model'];
        if (model) {
          await saveModelNode(model, modelDir);
          originalLog(`✅ Saved to ${modelDir}`);
        }
        
        // Resume progress updates
        setTimeout(() => {
          progressInterval.id = setInterval(updateProgress, 250);
        }, 2000);
      }
    }
    
  } finally {
    clearInterval(progressInterval);
    console.log = originalLog;
  }
  
  // Final save
  readline.cursorTo(process.stdout, 0, 25);
  console.log('\n✅ TRAINING COMPLETE!\n');
  
  const totalTime = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  console.log(`Total time: ${totalTime} minutes`);
  console.log(`Episodes completed: ${episodes}`);
  
  const finalDir = './models/yaniv-ai-model-final';
  if (!fs.existsSync(finalDir)) {
    fs.mkdirSync(finalDir, { recursive: true });
  }
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, finalDir);
    console.log(`\nFinal model saved to: ${finalDir}`);
  }
  
  console.log('\n🎉 Your AI is ready!\n');
  process.exit(0);
}

realtimeTraining().catch(console.error);