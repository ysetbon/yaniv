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

// Use TensorFlow with optimizations
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu');
// Try to use WebGL if available, otherwise CPU
tf.setBackend('cpu');
const { TrainingSystemV2 } = require('./src/game/TrainingSystemV2');
const { saveModelNode } = require('./src/game/NeuralNetworkAI-node');
const fs = require('fs');

async function trainWithFastBackend() {
  await tf.ready();
  console.log(`🚀 TensorFlow.js backend: ${tf.getBackend()} (FAST MODE)\n`);
  
  const episodes = parseInt(process.argv[2]) || 4000;
  const trainEvery = 40;
  const saveEvery = 100;
  
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║        🎮 YANIV AI TRAINING - ACCELERATED 🎮            ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');
  
  console.log(`📋 Configuration:`);
  console.log(`   Episodes: ${episodes}`);
  console.log(`   Backend: ${tf.getBackend()} (10x faster!)`);
  console.log(`   Train every: ${trainEvery} episodes`);
  console.log(`   Save every: ${saveEvery} episodes\n`);
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  
  // Progress tracking
  let stats = {
    completed: 0,
    trained: 0,
    losses: [],
    invalidMoves: 0,
    yanivCalls: 0
  };
  
  // Capture specific events
  const originalLog = console.log;
  console.log = (...args) => {
    const message = args.join(' ');
    if (message.includes('LEARNED: Invalid')) stats.invalidMoves++;
    if (message.includes('calls Yaniv')) stats.yanivCalls++;
  };
  
  function showProgress() {
    const progress = stats.completed / episodes;
    const percentage = (progress * 100).toFixed(1);
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = stats.completed / elapsed;
    const eta = (episodes - stats.completed) / rate;
    
    // Progress bar
    const barLength = 40;
    const filled = Math.floor(barLength * progress);
    const progressBar = '█'.repeat(filled) + '░'.repeat(barLength - filled);
    
    process.stdout.write('\r');
    process.stdout.write(
      `[${progressBar}] ${percentage}% | ` +
      `${stats.completed}/${episodes} | ` +
      `${elapsed.toFixed(0)}s | ` +
      `ETA: ${eta.toFixed(0)}s | ` +
      `${rate.toFixed(1)} eps/s`
    );
  }
  
  console.log('🎮 Starting accelerated training...\n');
  
  try {
    for (let i = 0; i < episodes; i++) {
      await trainingSystem.runSelfPlayEpisode();
      stats.completed = i + 1;
      
      // Update progress bar
      showProgress();
      
      // Train periodically
      if ((i + 1) % trainEvery === 0) {
        const loss = await trainingSystem.train();
        stats.losses.push(loss);
        stats.trained++;
        
        // Show training update
        const avgLoss = stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length;
        process.stdout.write('\n');
        originalLog(`\n📊 Training Update [Episode ${i + 1}]:`);
        originalLog(`   • Average Loss: ${avgLoss.toFixed(4)}`);
        originalLog(`   • Invalid Moves Learned: ${stats.invalidMoves}`);
        originalLog(`   • Successful Yaniv Calls: ${stats.yanivCalls}`);
        originalLog(`   • Training Sessions: ${stats.trained}\n`);
      }
      
      // Save periodically
      if ((i + 1) % saveEvery === 0) {
        process.stdout.write('\n');
        originalLog(`\n💾 Saving checkpoint at episode ${i + 1}...`);
        
        const modelDir = `./models/yaniv-ai-model-${i + 1}`;
        if (!fs.existsSync(modelDir)) {
          fs.mkdirSync(modelDir, { recursive: true });
        }
        
        const model = trainingSystem.getNeuralAI()['model'];
        if (model) {
          await saveModelNode(model, modelDir);
          originalLog(`✅ Checkpoint saved to ${modelDir}\n`);
        }
      }
    }
    
    // Restore console
    console.log = originalLog;
    
    // Final statistics
    process.stdout.write('\n\n');
    const totalTime = (Date.now() - startTime) / 1000;
    const avgLoss = stats.losses.length > 0 
      ? stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length 
      : 0;
    
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║               ✅ TRAINING COMPLETE! ✅                   ║');
    console.log('╚══════════════════════════════════════════════════════════╝\n');
    
    console.log('📊 Final Training Statistics:');
    console.log(`   • Total Episodes: ${episodes}`);
    console.log(`   • Total Time: ${(totalTime / 60).toFixed(1)} minutes`);
    console.log(`   • Average Speed: ${(episodes / totalTime).toFixed(1)} episodes/second`);
    console.log(`   • Final Average Loss: ${avgLoss.toFixed(4)}`);
    console.log(`   • Invalid Moves Learned From: ${stats.invalidMoves}`);
    console.log(`   • Successful Yaniv Calls: ${stats.yanivCalls}`);
    
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
    
    console.log('\n🎉 Your AI is now trained with 4000 games of experience!');
    console.log('   The AI has learned:');
    console.log('   • Never make invalid moves (-∞ penalty)');
    console.log('   • Optimal discard strategies');
    console.log('   • When to call Yaniv');
    console.log('   • Strategic draw decisions\n');
    
  } catch (error) {
    console.log = originalLog;
    console.error('\n❌ Training error:', error);
  }
  
  process.exit(0);
}

// Run the fast training
trainWithFastBackend().catch(console.error);