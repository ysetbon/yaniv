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

// Create readline interface for progress updates
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function clearLine() {
  readline.clearLine(process.stdout, 0);
  readline.cursorTo(process.stdout, 0);
}

function updateProgress(current, total, startTime, extraInfo = '') {
  const progress = (current / total * 100).toFixed(1);
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
  const rate = (current / elapsed).toFixed(1);
  const eta = ((total - current) / rate).toFixed(0);
  
  const barLength = 40;
  const filled = Math.floor(barLength * current / total);
  const bar = '█'.repeat(filled) + '░'.repeat(barLength - filled);
  
  clearLine();
  process.stdout.write(
    `Progress: [${bar}] ${progress}% | ` +
    `Episode ${current}/${total} | ` +
    `Time: ${elapsed}s | ` +
    `ETA: ${eta}s | ` +
    `Rate: ${rate} eps/s` +
    (extraInfo ? ` | ${extraInfo}` : '')
  );
}

async function runProgressTraining() {
  await tf.ready();
  console.log('TensorFlow.js backend:', tf.getBackend());
  console.log('=== Yaniv AI Training System ===\n');
  
  const episodes = 4000;
  const trainEvery = 40;
  const saveEvery = 100;
  
  console.log(`Configuration:`);
  console.log(`- Episodes: ${episodes}`);
  console.log(`- Train every: ${trainEvery} episodes`);
  console.log(`- Save every: ${saveEvery} episodes`);
  console.log(`- Experience buffer: 50,000 max\n`);
  
  console.log('Starting training...\n');
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  let totalLoss = 0;
  let lossCount = 0;
  
  // Suppress verbose logging during training
  const originalLog = console.log;
  const originalError = console.error;
  let messageBuffer = [];
  
  console.log = (...args) => {
    const message = args.join(' ');
    // Only capture important messages
    if (message.includes('Episode ended') || 
        message.includes('LEARNED') ||
        message.includes('calls Yaniv')) {
      messageBuffer.push(message);
    }
  };
  
  console.error = (...args) => {
    // Suppress error spam during training
  };
  
  try {
    for (let i = 0; i < episodes; i++) {
      try {
        // Run episode
        await trainingSystem.runSelfPlayEpisode();
        
        // Update progress bar
        updateProgress(i + 1, episodes, startTime);
        
        // Train periodically
        if ((i + 1) % trainEvery === 0) {
          const avgLoss = await trainingSystem.train();
          totalLoss += avgLoss;
          lossCount++;
          
          // Show training update
          clearLine();
          originalLog(`\n[Episode ${i + 1}] Training completed - Loss: ${avgLoss.toFixed(4)}`);
          
          // Show buffered messages
          if (messageBuffer.length > 0) {
            originalLog('Recent events:');
            messageBuffer.slice(-3).forEach(msg => originalLog(`  - ${msg}`));
            messageBuffer = [];
          }
          
          originalLog(''); // New line for progress bar
        }
        
        // Save periodically
        if ((i + 1) % saveEvery === 0) {
          clearLine();
          originalLog(`\n💾 Saving checkpoint at episode ${i + 1}...`);
          
          const modelDir = `./models/yaniv-ai-model-${i + 1}`;
          if (!fs.existsSync(modelDir)) {
            fs.mkdirSync(modelDir, { recursive: true });
          }
          
          const model = trainingSystem.getNeuralAI()['model'];
          if (model) {
            await saveModelNode(model, modelDir);
            originalLog(`✅ Model saved to ${modelDir}`);
          }
          originalLog(''); // New line for progress bar
        }
        
      } catch (episodeError) {
        // Continue training despite errors
      }
    }
    
    // Training complete
    clearLine();
    console.log = originalLog;
    console.error = originalError;
    
    // Final save
    console.log('\n\n🎯 Training completed! Saving final model...');
    
    const finalModelDir = './models/yaniv-ai-model-4000-final';
    if (!fs.existsSync(finalModelDir)) {
      fs.mkdirSync(finalModelDir, { recursive: true });
    }
    
    const model = trainingSystem.getNeuralAI()['model'];
    if (model) {
      await saveModelNode(model, finalModelDir);
    }
    
    const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
    const avgLossOverall = lossCount > 0 ? (totalLoss / lossCount).toFixed(4) : 'N/A';
    
    console.log('\n✅ Training Complete!\n');
    console.log('📊 Final Statistics:');
    console.log(`  - Total Episodes: ${episodes}`);
    console.log(`  - Total Time: ${(totalTime / 60).toFixed(1)} minutes`);
    console.log(`  - Average Loss: ${avgLossOverall}`);
    console.log(`  - Episodes/second: ${(episodes / totalTime).toFixed(2)}`);
    console.log(`\n📁 Final model saved to: ${finalModelDir}`);
    console.log('\n🚀 Your AI is now trained with 4000 games of experience!');
    console.log('   Copy the model files to your public/models directory to use in the game.');
    
  } catch (error) {
    console.log = originalLog;
    console.error = originalError;
    console.error('\n❌ Training failed:', error);
  }
  
  process.exit(0);
}

// Handle Ctrl+C gracefully
process.on('SIGINT', () => {
  console.log('\n\n⚠️  Training interrupted by user.');
  process.exit(0);
});

runProgressTraining().catch(console.error);