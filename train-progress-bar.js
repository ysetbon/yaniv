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

// Simple progress bar implementation
class ProgressBar {
  constructor(total, barLength = 40) {
    this.total = total;
    this.current = 0;
    this.barLength = barLength;
    this.startTime = Date.now();
    this.lastUpdate = Date.now();
  }

  update(current) {
    this.current = current;
    const now = Date.now();
    
    // Only update every 100ms to avoid flickering
    if (now - this.lastUpdate < 100 && current < this.total) {
      return;
    }
    this.lastUpdate = now;
    
    const progress = current / this.total;
    const filledLength = Math.floor(this.barLength * progress);
    const emptyLength = this.barLength - filledLength;
    
    const filledBar = '█'.repeat(filledLength);
    const emptyBar = '░'.repeat(emptyLength);
    const percentage = (progress * 100).toFixed(1);
    
    const elapsed = (now - this.startTime) / 1000;
    const rate = current / elapsed;
    const eta = rate > 0 ? (this.total - current) / rate : 0;
    
    // Clear line and write progress
    process.stdout.write('\r');
    process.stdout.write(
      `Progress: [${filledBar}${emptyBar}] ${percentage}% | ` +
      `${current}/${this.total} | ` +
      `Time: ${this.formatTime(elapsed)} | ` +
      `ETA: ${this.formatTime(eta)} | ` +
      `${rate.toFixed(1)} eps/s`
    );
  }

  formatTime(seconds) {
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}m ${secs}s`;
  }

  finish() {
    this.update(this.total);
    console.log('\n');
  }
}

async function trainWithProgressBar() {
  await tf.ready();
  console.log('TensorFlow.js backend:', tf.getBackend());
  console.log('\n🚀 Yaniv AI Training System\n');
  
  const episodes = parseInt(process.argv[2]) || 4000;
  const trainEvery = 40;
  const saveEvery = 100;
  
  console.log(`📋 Configuration:`);
  console.log(`   Episodes: ${episodes}`);
  console.log(`   Train every: ${trainEvery} episodes`);
  console.log(`   Save every: ${saveEvery} episodes`);
  console.log(`   AI Mode: Neural Network with Reinforcement Learning\n`);
  
  console.log(`🎮 Starting self-play training...\n`);
  
  const trainingSystem = new TrainingSystemV2();
  const progressBar = new ProgressBar(episodes);
  
  // Statistics tracking
  let stats = {
    totalLoss: 0,
    lossCount: 0,
    invalidMoves: 0,
    yanivCalls: 0,
    episodesWithErrors: 0
  };
  
  // Capture console output
  const originalLog = console.log;
  const originalError = console.error;
  let capturedMessages = [];
  
  console.log = (...args) => {
    const message = args.join(' ');
    if (message.includes('LEARNED: Invalid')) stats.invalidMoves++;
    if (message.includes('calls Yaniv')) stats.yanivCalls++;
    if (message.includes('Episode ended')) {
      capturedMessages.push(message);
    }
  };
  
  console.error = (...args) => {
    stats.episodesWithErrors++;
  };

  try {
    for (let i = 0; i < episodes; i++) {
      try {
        await trainingSystem.runSelfPlayEpisode();
        progressBar.update(i + 1);
        
        // Train periodically
        if ((i + 1) % trainEvery === 0) {
          const avgLoss = await trainingSystem.train();
          stats.totalLoss += avgLoss;
          stats.lossCount++;
          
          // Show training update
          process.stdout.write('\r\033[K'); // Clear line
          originalLog(`\n📊 [Episode ${i + 1}] Training update:`);
          originalLog(`   Loss: ${avgLoss.toFixed(4)} | Invalid moves learned: ${stats.invalidMoves} | Yaniv calls: ${stats.yanivCalls}`);
          
          // Show recent game results
          if (capturedMessages.length > 0) {
            const recent = capturedMessages.slice(-2);
            recent.forEach(msg => {
              const match = msg.match(/Winner: (.*?),/);
              if (match) originalLog(`   Recent game: ${msg.substring(msg.indexOf('Winner:'))}`);
            });
            capturedMessages = [];
          }
          console.log(''); // Empty line before progress bar
        }
        
        // Save periodically
        if ((i + 1) % saveEvery === 0) {
          process.stdout.write('\r\033[K'); // Clear line
          originalLog(`\n💾 Saving checkpoint...`);
          
          const modelDir = `./models/yaniv-ai-model-${i + 1}`;
          if (!fs.existsSync(modelDir)) {
            fs.mkdirSync(modelDir, { recursive: true });
          }
          
          const model = trainingSystem.getNeuralAI()['model'];
          if (model) {
            await saveModelNode(model, modelDir);
            originalLog(`✅ Model saved to ${modelDir}`);
          }
          console.log(''); // Empty line before progress bar
        }
        
      } catch (error) {
        // Continue despite errors
        stats.episodesWithErrors++;
      }
    }
    
    progressBar.finish();
    
  } finally {
    // Restore console
    console.log = originalLog;
    console.error = originalError;
  }
  
  // Final save
  console.log('💾 Saving final model...');
  const finalModelDir = './models/yaniv-ai-model-final';
  if (!fs.existsSync(finalModelDir)) {
    fs.mkdirSync(finalModelDir, { recursive: true });
  }
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, finalModelDir);
  }
  
  // Show final statistics
  const avgLoss = stats.lossCount > 0 ? (stats.totalLoss / stats.lossCount).toFixed(4) : 'N/A';
  
  console.log('\n✅ Training Complete!\n');
  console.log('📊 Final Statistics:');
  console.log(`   Episodes completed: ${episodes}`);
  console.log(`   Average loss: ${avgLoss}`);
  console.log(`   Invalid moves attempted: ${stats.invalidMoves}`);
  console.log(`   Successful Yaniv calls: ${stats.yanivCalls}`);
  console.log(`   Error rate: ${(stats.episodesWithErrors / episodes * 100).toFixed(1)}%`);
  console.log(`\n📁 Final model saved to: ${finalModelDir}`);
  console.log('\n🎉 Your AI is now trained and ready to play!');
  
  process.exit(0);
}

// Handle interruption
process.on('SIGINT', () => {
  console.log('\n\n⚠️  Training interrupted by user');
  process.exit(0);
});

// Run training
console.log('Starting Yaniv AI Training...\n');
trainWithProgressBar().catch(error => {
  console.error('\n❌ Training failed:', error);
  process.exit(1);
});