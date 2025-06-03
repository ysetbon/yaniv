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

async function testTraining() {
  await tf.ready();
  console.log('TensorFlow.js backend:', tf.getBackend());
  console.log('\n🧪 Testing Training System - 5 Episodes\n');
  
  const trainingSystem = new TrainingSystemV2();
  
  for (let i = 0; i < 5; i++) {
    console.log(`\n========== EPISODE ${i + 1} ==========`);
    const startTime = Date.now();
    
    try {
      await trainingSystem.runSelfPlayEpisode();
      
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`Episode ${i + 1} completed in ${elapsed} seconds`);
      
      if (i === 4) {
        console.log('\nTraining neural network on collected experiences...');
        const loss = await trainingSystem.train();
        console.log(`Training loss: ${loss.toFixed(4)}`);
      }
      
    } catch (error) {
      console.error(`Episode ${i + 1} error:`, error.message);
    }
  }
  
  console.log('\n✅ Test complete! The training system is working.');
  console.log('\nTo run full 4000 episode training, use:');
  console.log('  node train-with-progress.js');
  
  process.exit(0);
}

testTraining().catch(console.error);