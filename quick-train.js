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

async function runQuickTraining() {
  await tf.ready();
  console.log('TensorFlow.js backend:', tf.getBackend());
  console.log('\n🚀 Quick Training Mode - 100 Episodes Demo\n');
  
  const episodes = 100;  // Quick demo
  const trainEvery = 10;
  const saveEvery = 50;
  
  console.log(`Training Configuration:`);
  console.log(`- Episodes: ${episodes} (demo mode)`);
  console.log(`- Train every: ${trainEvery} episodes`);
  console.log(`- Save every: ${saveEvery} episodes\n`);
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  let invalidMoveAttempts = 0;
  let validMoves = 0;
  
  // Override console.log temporarily to capture specific logs
  const originalLog = console.log;
  console.log = (...args) => {
    const message = args.join(' ');
    if (message.includes('LEARNED: Invalid discard rejected')) {
      invalidMoveAttempts++;
    } else if (message.includes('discards') && message.includes('reward:')) {
      validMoves++;
    }
    
    // Only show important messages
    if (message.includes('Episode') || 
        message.includes('Loss:') || 
        message.includes('✓') ||
        message.includes('LEARNED') ||
        message.includes('Episode ended')) {
      originalLog(...args);
    }
  };
  
  console.log('🎮 Starting AI self-play training...\n');
  
  for (let i = 0; i < episodes; i++) {
    try {
      await trainingSystem.runSelfPlayEpisode();
      
      if ((i + 1) % trainEvery === 0) {
        const avgLoss = await trainingSystem.train();
        const progress = ((i + 1) / episodes * 100).toFixed(1);
        const elapsedTime = ((Date.now() - startTime) / 1000).toFixed(1);
        
        originalLog(`\n📊 Progress: ${progress}% | Episode ${i + 1}/${episodes}`);
        originalLog(`   Loss: ${avgLoss.toFixed(4)} | Time: ${elapsedTime}s`);
        originalLog(`   Invalid Moves Learned: ${invalidMoveAttempts} | Valid Moves: ${validMoves}`);
      }
      
      if ((i + 1) % saveEvery === 0) {
        const modelDir = `./models/yaniv-ai-checkpoint-${i + 1}`;
        if (!fs.existsSync(modelDir)) {
          fs.mkdirSync(modelDir, { recursive: true });
        }
        
        const model = trainingSystem.getNeuralAI()['model'];
        if (model) {
          await saveModelNode(model, modelDir);
          originalLog(`\n✅ Checkpoint saved: ${modelDir}`);
        }
      }
      
    } catch (error) {
      originalLog(`Episode ${i + 1} error:`, error.message);
    }
  }
  
  // Restore original console.log
  console.log = originalLog;
  
  // Final save
  const finalModelDir = './models/yaniv-ai-demo-100';
  if (!fs.existsSync(finalModelDir)) {
    fs.mkdirSync(finalModelDir, { recursive: true });
  }
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, finalModelDir);
  }
  
  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  
  console.log(`\n\n✅ Quick Training Complete!`);
  console.log(`\n📊 Training Summary:`);
  console.log(`  - Total Time: ${totalTime} seconds`);
  console.log(`  - Episodes: ${episodes}`);
  console.log(`  - Invalid Move Attempts: ${invalidMoveAttempts}`);
  console.log(`  - Valid Moves Executed: ${validMoves}`);
  console.log(`  - Learning Rate: ${(invalidMoveAttempts / validMoves * 100).toFixed(1)}% exploration`);
  console.log(`\n📁 Model saved to: ${finalModelDir}`);
  console.log(`\n💡 To train for 4000 episodes:`);
  console.log(`   - Estimated time: ${(totalTime / episodes * 4000 / 60).toFixed(0)} minutes`);
  console.log(`   - Run: node run-training-4000.js`);
  console.log(`   - Or use the interactive: npm run train`);
  
  process.exit(0);
}

runQuickTraining().catch(console.error);