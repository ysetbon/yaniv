/**
 * CLI Training Script for the Evolution → Distillation → RL Pipeline
 *
 * Run with: npx tsx src/train-evolution-rl.ts
 *
 * Phases:
 *  1. Genetic Evolution: Evolve 30 policies over 50 generations
 *  2. Distillation: Train NN to imitate top 5 elites
 *  3. PPO: Self-play fine-tuning to surpass genetic ceiling
 */

// Use Node.js TensorFlow backend for faster training
import '@tensorflow/tfjs-node';
import { EvolutionRLPipeline, PipelineProgress } from './game/evolution-rl/EvolutionRLPipeline';
import { NNSimPolicy } from './game/evolution-rl/PPOTrainer';
import { GeneticPolicy } from './game/evolution-rl/GeneticPolicy';
import { GameSimulator } from './game/evolution-rl/GameSimulator';

async function main() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('  Yaniv AI: Evolution → Distillation → RL Pipeline');
  console.log('═══════════════════════════════════════════════════════\n');

  const startTime = Date.now();

  // Parse CLI arguments for config overrides
  const args = process.argv.slice(2);
  const quick = args.includes('--quick');

  const config = quick ? {
    evolution: {
      populationSize: 15,
      generations: 20,
      gamesPerMatchup: 4,
      eliteCount: 3,
      matchupsPerPolicy: 5,
    },
    distillation: {
      gamesPerElite: 200,
      epochs: 15,
      batchSize: 32,
    },
    ppo: {
      selfPlayGamesPerBatch: 20,
      totalBatches: 15,
      ppoEpochs: 3,
    },
  } : {
    evolution: {
      populationSize: 30,
      generations: 50,
      gamesPerMatchup: 6,
      eliteCount: 5,
      matchupsPerPolicy: 8,
    },
    distillation: {
      gamesPerElite: 500,
      epochs: 30,
      batchSize: 64,
    },
    ppo: {
      selfPlayGamesPerBatch: 50,
      totalBatches: 40,
      ppoEpochs: 4,
    },
  };

  if (quick) {
    console.log('Running in QUICK mode (reduced parameters for testing)\n');
  }

  const pipeline = new EvolutionRLPipeline(config);

  let lastPhase = '';

  const model = await pipeline.run((progress: PipelineProgress) => {
    // Print phase transitions
    if (progress.phase !== lastPhase) {
      if (lastPhase) console.log('');
      console.log(`\n${'═'.repeat(55)}`);
      switch (progress.phase) {
        case 'evolution':
          console.log('  PHASE 1: GENETIC EVOLUTION');
          break;
        case 'distillation':
          console.log('  PHASE 2: DISTILLATION (Supervised Learning)');
          break;
        case 'ppo':
          console.log('  PHASE 3: RL FINE-TUNING (PPO Self-Play)');
          break;
        case 'done':
          console.log('  PIPELINE COMPLETE');
          break;
      }
      console.log(`${'═'.repeat(55)}\n`);
      lastPhase = progress.phase;
    }

    // Print progress
    const bar = progressBar(progress.phaseProgress, 30);
    console.log(`  ${bar} ${progress.message}`);
  });

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\nTotal training time: ${elapsed}s`);

  // Final validation
  console.log('\n═══════════════════════════════════════════════════════');
  console.log('  FINAL VALIDATION');
  console.log('═══════════════════════════════════════════════════════\n');

  const nnPolicy = new NNSimPolicy(model);
  const randomPolicy = new GeneticPolicy(); // random baseline

  // NN vs Random
  const vsRandom = GameSimulator.playMatch(nnPolicy, randomPolicy, 100);
  console.log(`  vs Random Policy:    ${vsRandom.wins0}/100 wins (${(vsRandom.wins0).toFixed(0)}%)`);

  // NN vs Elites
  const elites = pipeline.getElites();
  if (elites.length > 0) {
    let eliteWins = 0;
    let eliteGames = 0;
    for (const elite of elites) {
      const result = GameSimulator.playMatch(nnPolicy, elite, 50);
      eliteWins += result.wins0;
      eliteGames += 50;
    }
    console.log(`  vs Genetic Elites:   ${eliteWins}/${eliteGames} wins (${(eliteWins / eliteGames * 100).toFixed(1)}%)`);
  }

  // Save model
  try {
    const savePath = `file://./public/models/yaniv-evolution-rl`;
    await pipeline.saveModel(savePath);
    console.log(`\nModel saved to: public/models/yaniv-evolution-rl/`);
  } catch (err) {
    console.error('Failed to save model:', err);
  }

  console.log('\nDone!');
}

function progressBar(progress: number, width: number): string {
  const filled = Math.round(progress * width);
  const empty = width - filled;
  return `[${'#'.repeat(filled)}${'-'.repeat(empty)}] ${(progress * 100).toFixed(0).padStart(3)}%`;
}

main().catch(console.error);
