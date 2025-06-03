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

// ANSI escape codes for colors
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[36m',
  red: '\x1b[31m'
};

async function trainWithLiveProgress() {
  await tf.ready();
  
  console.clear();
  console.log(`${colors.blue}${colors.bright}╔══════════════════════════════════════════════════════════╗${colors.reset}`);
  console.log(`${colors.blue}${colors.bright}║           🎮 YANIV AI TRAINING SYSTEM 🎮                 ║${colors.reset}`);
  console.log(`${colors.blue}${colors.bright}╚══════════════════════════════════════════════════════════╝${colors.reset}\n`);
  
  const episodes = parseInt(process.argv[2]) || 4000;
  const trainEvery = 40;
  const saveEvery = 100;
  
  console.log(`${colors.yellow}📋 Configuration:${colors.reset}`);
  console.log(`   Episodes: ${colors.bright}${episodes}${colors.reset}`);
  console.log(`   Train every: ${trainEvery} episodes`);
  console.log(`   Save every: ${saveEvery} episodes`);
  console.log(`   Backend: ${tf.getBackend()}\n`);
  
  const trainingSystem = new TrainingSystemV2();
  const startTime = Date.now();
  
  // Initialize progress tracking
  let episodeCount = 0;
  let lastUpdateTime = Date.now();
  let recentTimes = [];
  let stats = {
    losses: [],
    invalidMoves: 0,
    yanivCalls: 0,
    gamesWon: 0
  };
  
  // Override console to capture game events
  const originalLog = console.log;
  const originalError = console.error;
  let capturedEvents = [];
  
  console.log = (...args) => {
    const message = args.join(' ');
    if (message.includes('LEARNED: Invalid')) {
      stats.invalidMoves++;
      capturedEvents.push(`❌ Invalid move learned: ${message.split('- ')[1]}`);
    }
    if (message.includes('calls Yaniv')) {
      stats.yanivCalls++;
      capturedEvents.push(`🎯 Yaniv called!`);
    }
    if (message.includes('Episode ended')) {
      const winner = message.match(/Winner: (.*?),/);
      if (winner) {
        stats.gamesWon++;
        capturedEvents.push(`🏆 Game ended - ${winner[0]}`);
      }
    }
  };
  
  console.error = () => {}; // Suppress errors
  
  // Progress display function
  function displayProgress(forceUpdate = false) {
    const now = Date.now();
    if (!forceUpdate && now - lastUpdateTime < 250) return; // Update every 250ms max
    lastUpdateTime = now;
    
    const elapsed = (now - startTime) / 1000;
    const progress = episodeCount / episodes;
    const percentage = (progress * 100).toFixed(1);
    
    // Calculate speed
    if (recentTimes.length > 10) recentTimes.shift();
    const avgTime = recentTimes.length > 0 
      ? recentTimes.reduce((a, b) => a + b, 0) / recentTimes.length 
      : 0;
    const speed = avgTime > 0 ? (1000 / avgTime).toFixed(1) : '...';
    const eta = episodeCount > 0 ? ((episodes - episodeCount) * avgTime / 1000).toFixed(0) : '...';
    
    // Progress bar
    const barWidth = 50;
    const filled = Math.floor(barWidth * progress);
    const empty = barWidth - filled;
    const bar = '█'.repeat(filled) + '░'.repeat(empty);
    
    // Clear screen and redraw
    console.clear();
    console.log(`${colors.blue}${colors.bright}╔══════════════════════════════════════════════════════════╗${colors.reset}`);
    console.log(`${colors.blue}${colors.bright}║           🎮 YANIV AI TRAINING SYSTEM 🎮                 ║${colors.reset}`);
    console.log(`${colors.blue}${colors.bright}╚══════════════════════════════════════════════════════════╝${colors.reset}\n`);
    
    // Progress bar
    console.log(`${colors.green}Progress: [${bar}] ${percentage}%${colors.reset}`);
    console.log(`Episodes: ${colors.bright}${episodeCount}/${episodes}${colors.reset} | Speed: ${colors.yellow}${speed} eps/s${colors.reset} | ETA: ${colors.blue}${eta}s${colors.reset}\n`);
    
    // Live statistics
    console.log(`${colors.yellow}📊 Live Statistics:${colors.reset}`);
    console.log(`   • Games Completed: ${stats.gamesWon}`);
    console.log(`   • Invalid Moves Learned: ${colors.red}${stats.invalidMoves}${colors.reset}`);
    console.log(`   • Yaniv Calls: ${colors.green}${stats.yanivCalls}${colors.reset}`);
    if (stats.losses.length > 0) {
      const avgLoss = stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length;
      console.log(`   • Average Loss: ${avgLoss.toFixed(4)}`);
    }
    
    // Recent events
    if (capturedEvents.length > 0) {
      console.log(`\n${colors.blue}📰 Recent Events:${colors.reset}`);
      capturedEvents.slice(-5).forEach(event => {
        console.log(`   ${event}`);
      });
    }
    
    // Training status
    if (episodeCount === 0) {
      console.log(`\n${colors.yellow}⏳ Initializing first episode...${colors.reset}`);
    } else if (episodeCount < 10) {
      console.log(`\n${colors.yellow}🚀 Warming up... (${episodeCount}/10 episodes)${colors.reset}`);
    } else {
      console.log(`\n${colors.green}✅ Training in progress...${colors.reset}`);
    }
  }
  
  // Initial display
  displayProgress(true);
  
  // Update display regularly
  const displayInterval = setInterval(() => displayProgress(true), 500);
  
  // Main training loop
  originalLog('\n🎮 Starting neural network training...\n');
  
  try {
    for (let i = 0; i < episodes; i++) {
      const episodeStart = Date.now();
      
      // Run episode
      await trainingSystem.runSelfPlayEpisode();
      
      // Track timing
      const episodeTime = Date.now() - episodeStart;
      recentTimes.push(episodeTime);
      episodeCount = i + 1;
      
      // Update display
      displayProgress();
      
      // Train periodically
      if ((i + 1) % trainEvery === 0) {
        displayProgress(true);
        const trainStart = Date.now();
        const loss = await trainingSystem.train();
        stats.losses.push(loss);
        
        // Clear recent events after training
        if (capturedEvents.length > 10) {
          capturedEvents = capturedEvents.slice(-5);
        }
      }
      
      // Save periodically
      if ((i + 1) % saveEvery === 0) {
        clearInterval(displayInterval);
        console.clear();
        originalLog(`\n${colors.green}💾 Saving checkpoint at episode ${i + 1}...${colors.reset}`);
        
        const modelDir = `./models/yaniv-ai-model-${i + 1}`;
        if (!fs.existsSync(modelDir)) {
          fs.mkdirSync(modelDir, { recursive: true });
        }
        
        const model = trainingSystem.getNeuralAI()['model'];
        if (model) {
          await saveModelNode(model, modelDir);
          originalLog(`${colors.green}✅ Model saved to ${modelDir}${colors.reset}`);
        }
        
        // Resume display updates
        setTimeout(() => {
          displayProgress(true);
          const newInterval = setInterval(() => displayProgress(true), 500);
          displayInterval.id = newInterval;
        }, 2000);
      }
    }
    
  } finally {
    clearInterval(displayInterval);
    console.log = originalLog;
    console.error = originalError;
  }
  
  // Training complete
  console.clear();
  console.log(`${colors.green}${colors.bright}╔══════════════════════════════════════════════════════════╗${colors.reset}`);
  console.log(`${colors.green}${colors.bright}║               ✅ TRAINING COMPLETE! ✅                   ║${colors.reset}`);
  console.log(`${colors.green}${colors.bright}╚══════════════════════════════════════════════════════════╝${colors.reset}\n`);
  
  const totalTime = (Date.now() - startTime) / 1000;
  const avgLoss = stats.losses.length > 0 
    ? stats.losses.reduce((a, b) => a + b, 0) / stats.losses.length 
    : 0;
  
  console.log(`${colors.yellow}📊 Final Training Statistics:${colors.reset}`);
  console.log(`   • Total Episodes: ${colors.bright}${episodes}${colors.reset}`);
  console.log(`   • Total Time: ${colors.bright}${(totalTime / 60).toFixed(1)} minutes${colors.reset}`);
  console.log(`   • Average Speed: ${(episodes / totalTime).toFixed(2)} episodes/second`);
  console.log(`   • Final Average Loss: ${avgLoss.toFixed(4)}`);
  console.log(`   • Games Completed: ${stats.gamesWon}`);
  console.log(`   • Invalid Moves Learned From: ${colors.red}${stats.invalidMoves}${colors.reset}`);
  console.log(`   • Successful Yaniv Calls: ${colors.green}${stats.yanivCalls}${colors.reset}`);
  
  // Save final model
  const finalDir = './models/yaniv-ai-model-final';
  if (!fs.existsSync(finalDir)) {
    fs.mkdirSync(finalDir, { recursive: true });
  }
  
  const model = trainingSystem.getNeuralAI()['model'];
  if (model) {
    await saveModelNode(model, finalDir);
    console.log(`\n${colors.green}📁 Final model saved to: ${finalDir}${colors.reset}`);
  }
  
  console.log(`\n${colors.green}${colors.bright}🎉 Your AI is now trained with ${episodes} games of experience!${colors.reset}`);
  console.log('\nThe AI has learned:');
  console.log('   • Never make invalid moves');
  console.log('   • Optimal card combinations');
  console.log('   • Strategic Yaniv timing');
  console.log('   • Smart draw decisions\n');
  
  process.exit(0);
}

// Handle Ctrl+C
process.on('SIGINT', () => {
  console.log('\n\n⚠️  Training interrupted by user');
  process.exit(0);
});

// Start training
trainWithLiveProgress().catch(error => {
  console.error('\n❌ Training error:', error);
  process.exit(1);
});