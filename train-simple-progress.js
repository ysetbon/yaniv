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

async function simpleProgressTraining() {
  await tf.ready();
  
  console.log('\n🎮 YANIV AI TRAINING SYSTEM\n');
  
  const episodes = parseInt(process.argv[2]) || 4000;
  const trainEvery = 40;
  const saveEvery = 100;
  
  console.log(`Configuration:`);
  console.log(`• Episodes: ${episodes}`);
  console.log(`• Backend: ${tf.getBackend()}`);
  console.log(`• Train every: ${trainEvery} episodes`);
  console.log(`• Save every: ${saveEvery} episodes\n`);
  
  console.log('Starting training... (first episode may take 10-30 seconds)\n');
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  let episodeCount = 0;
  let stats = {
    losses: [],
    invalidMoves: 0,
    yanivCalls: 0
  };
  
  // Simple spinner
  const spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  let spinIndex = 0;
  
  // Show spinner while processing
  let spinInterval = setInterval(() => {
    process.stdout.write(`\rProcessing episode ${episodeCount + 1}... ${spinner[spinIndex]} `);
    spinIndex = (spinIndex + 1) % spinner.length;
  }, 100);
  
  // Capture game events
  const originalLog = console.log;
  console.log = (...args) => {
    const message = args.join(' ');
    if (message.includes('LEARNED: Invalid')) stats.invalidMoves++;
    if (message.includes('calls Yaniv')) stats.yanivCalls++;
  };
  
  try {
    // Training loop
    for (let i = 0; i < episodes; i++) {
      const episodeStart = Date.now();
      
      // Run episode
      await trainingSystem.runSelfPlayEpisode();
      episodeCount = i + 1;
      
      // Calculate timing
      const episodeTime = ((Date.now() - episodeStart) / 1000).toFixed(1);
      const totalTime = ((Date.now() - startTime) / 1000).toFixed(0);
      const avgTime = (totalTime / episodeCount).toFixed(1);
      const eta = ((episodes - episodeCount) * avgTime).toFixed(0);
      
      // Update progress line
      clearInterval(spinInterval);
      process.stdout.write('\r' + ' '.repeat(80) + '\r'); // Clear line
      
      const progress = (episodeCount / episodes * 100).toFixed(1);
      const barLength = 30;
      const filled = Math.floor(barLength * episodeCount / episodes);
      const bar = '█'.repeat(filled) + '░'.repeat(barLength - filled);
      
      process.stdout.write(
        `[${bar}] ${progress}% | Episode ${episodeCount}/${episodes} | ${avgTime}s/ep | ETA: ${eta}s`
      );
      
      // Restart spinner
      spinInterval = setInterval(() => {
        process.stdout.write(` ${spinner[spinIndex]}`);
        spinIndex = (spinIndex + 1) % spinner.length;
      }, 100);
      
      // Train periodically
      if (episodeCount % trainEvery === 0) {
        clearInterval(spinInterval);
        process.stdout.write('\n');
        originalLog(`\n📊 Training at episode ${episodeCount}...`);
        
        const loss = await trainingSystem.train();
        stats.losses.push(loss);
        
        originalLog(`   Loss: ${loss.toFixed(4)}`);
        originalLog(`   Invalid moves learned: ${stats.invalidMoves}`);
        originalLog(`   Yaniv calls: ${stats.yanivCalls}\n`);
        
        // Restart progress display
        spinInterval = setInterval(() => {
          process.stdout.write(`\rProcessing episode ${episodeCount + 1}... ${spinner[spinIndex]} `);
          spinIndex = (spinIndex + 1) % spinner.length;
        }, 100);
      }
      
      // Save periodically
      if (episodeCount % saveEvery === 0) {
        clearInterval(spinInterval);
        process.stdout.write('\n');
        originalLog(`\n💾 Saving model at episode ${episodeCount}...`);
        
        const modelDir = `./models/yaniv-ai-model-${episodeCount}`;
        if (!fs.existsSync(modelDir)) {
          fs.mkdirSync(modelDir, { recursive: true });
        }
        
        const model = trainingSystem.getNeuralAI()['model'];
        if (model) {
          await saveModelNode(model, modelDir);
          originalLog(`✅ Saved to ${modelDir}\n`);
        }
        
        // Restart spinner
        spinInterval = setInterval(() => {
          process.stdout.write(`\rProcessing episode ${episodeCount + 1}... ${spinner[spinIndex]} `);
          spinIndex = (spinIndex + 1) % spinner.length;
        }, 100);
      }
    }
    
  } finally {
    clearInterval(spinInterval);
    console.log = originalLog;
  }
  
  // Training complete
  process.stdout.write('\n\n');
  console.log('✅ TRAINING COMPLETE!\n');
  
  const totalTime = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
  const avgLoss = stats.losses.length > 0 
    ? stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length 
    : 0;
  
  console.log('📊 Final Statistics:');
  console.log(`   • Episodes: ${episodes}`);
  console.log(`   • Total time: ${totalTime} minutes`);
  console.log(`   • Average loss: ${avgLoss.toFixed(4)}`);
  console.log(`   • Invalid moves learned: ${stats.invalidMoves}`);
  console.log(`   • Yaniv calls: ${stats.yanivCalls}`);
  
  // Save final model
  const finalDir = './models/yaniv-ai-model-final';
  if (!fs.existsSync(finalDir)) {
    fs.mkdirSync(finalDir, { recursive: true });
  }
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, finalDir);
    console.log(`\n📁 Final model saved to: ${finalDir}`);
  }
  
  console.log('\n🎉 Your AI is ready to play!\n');
  process.exit(0);
}

// Handle Ctrl+C
process.on('SIGINT', () => {
  console.log('\n\nTraining interrupted by user.\n');
  process.exit(0);
});

// Start
console.log('============================');
console.log('Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details.');
console.log('============================');

simpleProgressTraining().catch(error => {
  console.error('\nError:', error);
  process.exit(1);
});