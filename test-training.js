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
  console.log('🧪 TESTING YANIV AI TRAINING SYSTEM\n');
  
  console.log('1. Initializing TensorFlow...');
  await tf.ready();
  console.log('   ✅ TensorFlow ready, backend:', tf.getBackend());
  
  console.log('\n2. Creating training system...');
  const trainingSystem = new TrainingSystemV2();
  console.log('   ✅ Training system created');
  
  console.log('\n3. Running ONE training episode...');
  console.log('   (This simulates a complete Yaniv game)\n');
  
  const startTime = Date.now();
  let stepCount = 0;
  
  // Override console to show game progress
  const originalLog = console.log;
  console.log = (...args) => {
    const message = args.join(' ');
    if (message.includes('Turn') || 
        message.includes('Episode ended') || 
        message.includes('LEARNED') ||
        message.includes('calls Yaniv')) {
      originalLog(`   [Step ${++stepCount}] ${message}`);
    }
  };
  
  try {
    console.log('   Starting game simulation...');
    await trainingSystem.runSelfPlayEpisode();
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log = originalLog;
    
    console.log(`\n   ✅ Episode completed in ${elapsed} seconds`);
    console.log(`   Total game steps: ${stepCount}`);
    
    console.log('\n4. Training neural network on collected data...');
    const loss = await trainingSystem.train();
    console.log(`   ✅ Training complete, loss: ${loss.toFixed(4)}`);
    
    console.log('\n🎉 SUCCESS! The training system is working correctly.');
    console.log('\n📊 Performance estimate:');
    console.log(`   • One episode took: ${elapsed} seconds`);
    console.log(`   • Estimated time for 100 episodes: ${(elapsed * 100 / 60).toFixed(1)} minutes`);
    console.log(`   • Estimated time for 4000 episodes: ${(elapsed * 4000 / 3600).toFixed(1)} hours`);
    
    console.log('\n💡 To run full training with progress:');
    console.log('   node train-simple-progress.js 100    (for quick test)');
    console.log('   node train-simple-progress.js 4000   (for full training)');
    
  } catch (error) {
    console.log = originalLog;
    console.error('\n❌ Error during training:', error);
    console.error('\nStack trace:', error.stack);
  }
  
  process.exit(0);
}

testTraining().catch(console.error);