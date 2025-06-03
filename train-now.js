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

async function trainNow() {
  // Initialize TensorFlow first
  await tf.ready();
  
  console.log('\n🎮 YANIV AI TRAINING\n');
  
  const episodes = parseInt(process.argv[2]) || 10;
  console.log(`Training ${episodes} episodes...`);
  console.log('Each * = 5 game turns\n');
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  
  let completed = 0;
  let turnCount = 0;
  let totalTurns = 0;
  let stats = { invalid: 0, yaniv: 0 };
  
  // Override console to track progress
  const originalLog = console.log;
  let lastProgressUpdate = Date.now();
  
  console.log = (...args) => {
    const msg = args.join(' ');
    if (msg.includes('Turn')) {
      turnCount++;
      totalTurns++;
      if (turnCount % 5 === 0) {
        process.stdout.write('*');
      }
      // Update progress more frequently
      if (Date.now() - lastProgressUpdate > 100) {
        process.stdout.write('');
        lastProgressUpdate = Date.now();
      }
    }
    if (msg.includes('LEARNED: Invalid')) stats.invalid++;
    if (msg.includes('calls Yaniv')) stats.yaniv++;
  };
  
  // Start training
  console.log('Starting...\n');
  process.stdout.write('Episode 1: ');
  
  try {
    for (let i = 0; i < episodes; i++) {
      // Add timeout to detect stuck episodes
      const episodeStart = Date.now();
      let progressTimer = setInterval(() => {
        if (turnCount > 0 && turnCount % 10 === 0) {
          process.stdout.write('.');
        }
      }, 500);
      
      try {
        await trainingSystem.runSelfPlayEpisode();
      } finally {
        clearInterval(progressTimer);
      }
      
      // Episode complete
      completed++;
      const time = ((Date.now() - startTime) / 1000).toFixed(1);
      const episodeTime = ((Date.now() - episodeStart) / 1000).toFixed(1);
      process.stdout.write(`\nEpisode ${completed}/${episodes} done (${turnCount} turns, ${episodeTime}s, total: ${time}s)\n`);
      turnCount = 0;
      
      // Train every 5 episodes
      if ((i + 1) % 5 === 0) {
        process.stdout.write('\nTraining network...');
        const loss = await trainingSystem.train();
        process.stdout.write(` Loss: ${loss.toFixed(4)}\n`);
      }
      
      // Start next episode
      if (i + 1 < episodes) {
        process.stdout.write(`\nEpisode ${i + 2}: `);
      }
    }
  } finally {
    console.log = originalLog;
  }
  
  // Summary
  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log('\n✅ COMPLETE!\n');
  console.log(`Time: ${totalTime}s (${(totalTime/episodes).toFixed(1)}s per episode)`);
  console.log(`Total turns: ${totalTurns}`);
  console.log(`Invalid moves learned: ${stats.invalid}`);
  console.log(`Yaniv calls: ${stats.yaniv}`);
  
  // Save
  const dir = './models/yaniv-trained-' + episodes;
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, dir);
    console.log(`\nModel saved: ${dir}`);
  }
  
  console.log(`\nFor 4000 episodes: ~${(totalTime/episodes*4000/3600).toFixed(1)} hours`);
  process.exit(0);
}

// Start immediately
console.log('Loading TensorFlow...');
trainNow().catch(e => {
  console.error('Error:', e.message);
  process.exit(1);
});