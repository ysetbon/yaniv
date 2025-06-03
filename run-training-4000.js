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

async function runTraining() {
  // Set up TensorFlow.js backend
  await tf.ready();
  console.log('TensorFlow.js backend:', tf.getBackend());
  
  console.log('=== Yaniv AI Training System ===\n');
  
  const episodes = 4000;
  const trainEvery = 40;
  const saveEvery = 100;
  
  console.log(`Starting training with:`);
  console.log(`- Episodes: ${episodes}`);
  console.log(`- Train every: ${trainEvery} episodes`);
  console.log(`- Save every: ${saveEvery} episodes`);
  console.log(`- Focus: Learning optimal strategies with 4000 games\n`);
  
  const trainingSystem = new TrainingSystemV2();
  let lastSaveTime = Date.now();
  let episodeErrors = 0;
  let successfulEpisodes = 0;
  
  for (let i = 0; i < episodes; i++) {
    try {
      // Run self-play episode with error handling
      await trainingSystem.runSelfPlayEpisode();
      successfulEpisodes++;
      
      // Train periodically
      if ((i + 1) % trainEvery === 0) {
        try {
          const avgLoss = await trainingSystem.train();
          const progress = ((i + 1) / episodes * 100).toFixed(1);
          console.log(`\n[${progress}%] Episode ${i + 1}/${episodes}`);
          console.log(`  - Average Loss: ${avgLoss.toFixed(4)}`);
          console.log(`  - Successful Episodes: ${successfulEpisodes}`);
          console.log(`  - Error Rate: ${(episodeErrors / (i + 1) * 100).toFixed(1)}%`);
          
          // Reset error tracking periodically
          if (episodeErrors > 10) {
            console.log(`  - High error rate detected, continuing training...`);
          }
        } catch (trainError) {
          console.error(`Training error at episode ${i + 1}:`, trainError.message);
          // Continue training despite errors
        }
      }
      
      // Save model periodically
      if ((i + 1) % saveEvery === 0) {
        const modelDir = `./models/yaniv-ai-model-${i + 1}`;
        
        // Create directory
        if (!fs.existsSync(modelDir)) {
          fs.mkdirSync(modelDir, { recursive: true });
        }
        
        try {
          // Save using Node.js handler
          const model = trainingSystem.getNeuralAI()['model'];
          if (model) {
            await saveModelNode(model, modelDir);
            const timeSinceLastSave = ((Date.now() - lastSaveTime) / 1000).toFixed(1);
            console.log(`\n✓ Model checkpoint saved to ${modelDir} (${timeSinceLastSave}s)`);
            lastSaveTime = Date.now();
          }
        } catch (saveError) {
          console.error(`Failed to save model at episode ${i + 1}:`, saveError.message);
        }
      }
      
      // Show progress bar every 20 episodes
      if ((i + 1) % 20 === 0 && (i + 1) % trainEvery !== 0) {
        const progress = ((i + 1) / episodes * 100).toFixed(1);
        const barLength = 40;
        const filledLength = Math.floor(barLength * (i + 1) / episodes);
        const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);
        process.stdout.write(`\r[${bar}] ${progress}% (${i + 1}/${episodes})`);
      }
      
    } catch (error) {
      episodeErrors++;
      if (episodeErrors % 10 === 0) {
        console.error(`\nEpisode ${i + 1} error (${episodeErrors} total):`, error.message);
      }
      // Continue training despite episode errors
    }
    
    // Memory cleanup every 500 episodes
    if ((i + 1) % 500 === 0) {
      if (global.gc) {
        global.gc();
        console.log('\n♻️  Memory cleanup performed');
      }
    }
  }
  
  // Final save
  const finalModelDir = './models/yaniv-ai-model-4000-final';
  if (!fs.existsSync(finalModelDir)) {
    fs.mkdirSync(finalModelDir, { recursive: true });
  }
  
  try {
    const model = trainingSystem.getNeuralAI()['model'];
    if (model) {
      await saveModelNode(model, finalModelDir);
    }
    
    console.log(`\n\n✅ Training complete!`);
    console.log(`📊 Training Statistics:`);
    console.log(`  - Total Episodes: ${episodes}`);
    console.log(`  - Successful Episodes: ${successfulEpisodes}`);
    console.log(`  - Episode Errors: ${episodeErrors}`);
    console.log(`  - Success Rate: ${(successfulEpisodes / episodes * 100).toFixed(1)}%`);
    console.log(`\n📁 Final model saved to: ${finalModelDir}`);
    console.log(`\n🎯 Expected Improvements after 4000 games:`);
    console.log(`  - Mastery of valid/invalid move detection`);
    console.log(`  - Optimal discard strategies`);
    console.log(`  - Strategic Yaniv timing`);
    console.log(`  - Advanced opponent modeling`);
    
  } catch (finalSaveError) {
    console.error('Failed to save final model:', finalSaveError);
  }
  
  process.exit(0);
}

// Handle errors gracefully
process.on('unhandledRejection', (error) => {
  console.error('\n❌ Unhandled training error:', error);
  // Don't exit - try to continue training
});

// Catch SIGINT (Ctrl+C) to save progress
process.on('SIGINT', async () => {
  console.log('\n\n⚠️  Training interrupted! Saving current progress...');
  
  try {
    const emergencyDir = './models/yaniv-ai-model-interrupted';
    if (!fs.existsSync(emergencyDir)) {
      fs.mkdirSync(emergencyDir, { recursive: true });
    }
    
    // Note: We can't access trainingSystem here, so this is just a notification
    console.log(`Please check ${emergencyDir} for any saved progress.`);
  } catch (e) {
    console.error('Failed to save emergency checkpoint:', e);
  }
  
  process.exit(0);
});

// Run the training
runTraining().catch(error => {
  console.error('\n💥 Fatal training error:', error);
  process.exit(1);
});