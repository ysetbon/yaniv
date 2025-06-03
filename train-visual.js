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

async function visualTraining() {
  await tf.ready();
  
  console.clear();
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║           🎮 YANIV AI TRAINING SYSTEM 🎮                 ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');
  
  const episodes = parseInt(process.argv[2]) || 100;
  console.log(`📊 Training Configuration:`);
  console.log(`   • Episodes: ${episodes}`);
  console.log(`   • Backend: ${tf.getBackend()}`);
  console.log(`   • Learning: Reinforcement Learning with -∞ penalties\n`);
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  
  // Visual elements
  const spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  let spinnerIndex = 0;
  
  // Stats
  let stats = {
    completed: 0,
    invalidMoves: 0,
    yanivCalls: 0,
    losses: []
  };
  
  // Override console to capture events
  const originalLog = console.log;
  console.log = (...args) => {
    const message = args.join(' ');
    if (message.includes('LEARNED: Invalid')) stats.invalidMoves++;
    if (message.includes('calls Yaniv')) stats.yanivCalls++;
  };
  
  // Update display function
  function updateDisplay() {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
    const progress = stats.completed / episodes;
    const percentage = (progress * 100).toFixed(1);
    
    // Progress bar
    const barLength = 50;
    const filled = Math.floor(barLength * progress);
    const empty = barLength - filled;
    const progressBar = '█'.repeat(filled) + '░'.repeat(empty);
    
    // Clear and redraw
    process.stdout.write('\033[2J\033[0f'); // Clear screen and move to top
    
    console.log('╔══════════════════════════════════════════════════════════╗');
    console.log('║           🎮 YANIV AI TRAINING SYSTEM 🎮                 ║');
    console.log('╚══════════════════════════════════════════════════════════╝\n');
    
    console.log(`Progress: [${progressBar}] ${percentage}%`);
    console.log(`Episodes: ${stats.completed}/${episodes} ${spinner[spinnerIndex]}`);
    console.log(`Time: ${elapsed}s | Rate: ${(stats.completed / elapsed).toFixed(1)} eps/s\n`);
    
    console.log('📊 Live Statistics:');
    console.log(`   • Invalid Moves Learned: ${stats.invalidMoves}`);
    console.log(`   • Yaniv Calls: ${stats.yanivCalls}`);
    console.log(`   • Avg Loss: ${stats.losses.length > 0 ? (stats.losses.reduce((a,b) => a+b, 0) / stats.losses.length).toFixed(4) : 'N/A'}`);
    
    if (stats.completed === episodes) {
      console.log('\n✅ Training Complete!');
    }
    
    spinnerIndex = (spinnerIndex + 1) % spinner.length;
  }
  
  // Initial display
  updateDisplay();
  
  // Update display every 500ms
  const displayInterval = setInterval(updateDisplay, 500);
  
  try {
    for (let i = 0; i < episodes; i++) {
      await trainingSystem.runSelfPlayEpisode();
      stats.completed = i + 1;
      
      // Train every 10 episodes for faster feedback
      if ((i + 1) % 10 === 0) {
        const loss = await trainingSystem.train();
        stats.losses.push(loss);
      }
      
      // Save every 50 episodes
      if ((i + 1) % 50 === 0) {
        const modelDir = `./models/yaniv-ai-checkpoint-${i + 1}`;
        if (!fs.existsSync(modelDir)) {
          fs.mkdirSync(modelDir, { recursive: true });
        }
        
        const model = trainingSystem.getNeuralAI()['model'];
        if (model) {
          await saveModelNode(model, modelDir);
        }
      }
    }
    
  } finally {
    clearInterval(displayInterval);
    console.log = originalLog;
  }
  
  // Final update
  updateDisplay();
  
  // Save final model
  const finalDir = './models/yaniv-ai-visual-trained';
  if (!fs.existsSync(finalDir)) {
    fs.mkdirSync(finalDir, { recursive: true });
  }
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, finalDir);
    console.log(`\n📁 Model saved to: ${finalDir}`);
  }
  
  console.log('\n🎉 Training complete! Your AI is ready to play Yaniv!\n');
  process.exit(0);
}

visualTraining().catch(error => {
  console.error('\n❌ Error:', error);
  process.exit(1);
});