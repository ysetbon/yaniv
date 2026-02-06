/**
 * EvolutionRLPipeline: Orchestrates the three-phase training process.
 *
 *  Phase 1: Genetic Evolution
 *    → Evolve 30 policies over 50 generations
 *    → Extract top 5 elite heuristic policies
 *
 *  Phase 2: Distillation
 *    → Run elites through thousands of games
 *    → Record every (state, action) pair
 *    → Train neural network via supervised learning
 *    → NN now plays ~as well as genetic elites
 *
 *  Phase 3: RL Fine-tuning (PPO)
 *    → Self-play with PPO starting from distilled NN
 *    → NN discovers strategies beyond heuristic ceiling
 *    → Surpasses the genetic elite
 */

import * as tf from '@tensorflow/tfjs';
import {
  GeneticEvolution, EvolutionConfig, DEFAULT_EVOLUTION_CONFIG,
  GenerationStats,
} from './GeneticEvolution';
import {
  DistillationTrainer, DistillationConfig, DEFAULT_DISTILLATION_CONFIG,
  DistillationStats,
} from './DistillationTrainer';
import {
  PPOTrainer, PPOConfig, DEFAULT_PPO_CONFIG,
  PPOStats, NNSimPolicy,
} from './PPOTrainer';
import { GeneticPolicy } from './GeneticPolicy';
import { GameSimulator } from './GameSimulator';

export interface PipelineConfig {
  evolution: Partial<EvolutionConfig>;
  distillation: Partial<DistillationConfig>;
  ppo: Partial<PPOConfig>;
}

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  evolution: DEFAULT_EVOLUTION_CONFIG,
  distillation: DEFAULT_DISTILLATION_CONFIG,
  ppo: DEFAULT_PPO_CONFIG,
};

export type PipelinePhase = 'idle' | 'evolution' | 'distillation' | 'ppo' | 'done';

export interface PipelineProgress {
  phase: PipelinePhase;
  phaseProgress: number;  // 0-1
  message: string;
  // Phase-specific data
  evolutionStats?: GenerationStats;
  distillationStats?: DistillationStats;
  ppoStats?: PPOStats;
}

export type PipelineProgressCallback = (progress: PipelineProgress) => void;

export class EvolutionRLPipeline {
  private config: PipelineConfig;
  private phase: PipelinePhase = 'idle';
  private elites: GeneticPolicy[] = [];
  private distillationTrainer: DistillationTrainer;

  constructor(config?: Partial<PipelineConfig>) {
    this.config = {
      evolution: { ...DEFAULT_EVOLUTION_CONFIG, ...config?.evolution },
      distillation: { ...DEFAULT_DISTILLATION_CONFIG, ...config?.distillation },
      ppo: { ...DEFAULT_PPO_CONFIG, ...config?.ppo },
    };
    this.distillationTrainer = new DistillationTrainer(this.config.distillation);
  }

  /**
   * Run the full three-phase pipeline.
   */
  async run(onProgress?: PipelineProgressCallback): Promise<tf.LayersModel> {
    // ══════════════════════════════════════════
    //  PHASE 1: GENETIC EVOLUTION
    // ══════════════════════════════════════════
    this.phase = 'evolution';
    onProgress?.({
      phase: 'evolution',
      phaseProgress: 0,
      message: 'Phase 1: Starting genetic evolution...',
    });

    const evolution = new GeneticEvolution(this.config.evolution);
    const totalGens = this.config.evolution.generations || DEFAULT_EVOLUTION_CONFIG.generations;

    this.elites = evolution.evolve((stats: GenerationStats) => {
      onProgress?.({
        phase: 'evolution',
        phaseProgress: stats.generation / totalGens,
        message: `Gen ${stats.generation}/${totalGens} | Best: ${(stats.bestFitness * 100).toFixed(1)}% | Avg: ${(stats.avgFitness * 100).toFixed(1)}%`,
        evolutionStats: stats,
      });
    });

    onProgress?.({
      phase: 'evolution',
      phaseProgress: 1,
      message: `Phase 1 complete. ${this.elites.length} elites selected (best: ${(this.elites[0].fitness * 100).toFixed(1)}% win rate)`,
    });

    // ══════════════════════════════════════════
    //  PHASE 2: DISTILLATION
    // ══════════════════════════════════════════
    this.phase = 'distillation';
    onProgress?.({
      phase: 'distillation',
      phaseProgress: 0,
      message: 'Phase 2: Collecting training data from elites...',
    });

    // Collect (state, action) pairs from elite play
    const trainingData = this.distillationTrainer.collectTrainingData(this.elites);

    onProgress?.({
      phase: 'distillation',
      phaseProgress: 0.3,
      message: `Collected ${trainingData.length} training samples. Building neural network...`,
    });

    // Build and train the neural network
    this.distillationTrainer.buildModel();
    const totalEpochs = this.config.distillation.epochs || DEFAULT_DISTILLATION_CONFIG.epochs;

    await this.distillationTrainer.train(trainingData, (stats: DistillationStats) => {
      const progress = 0.3 + 0.7 * (stats.epoch / totalEpochs);
      onProgress?.({
        phase: 'distillation',
        phaseProgress: progress,
        message: `Epoch ${stats.epoch}/${totalEpochs} | Discard loss: ${stats.discardLoss.toFixed(4)} | Draw loss: ${stats.drawLoss.toFixed(4)}`,
        distillationStats: stats,
      });
    });

    // Validate distilled model against elites
    const distilledWinRate = await this.validateDistilledModel();

    onProgress?.({
      phase: 'distillation',
      phaseProgress: 1,
      message: `Phase 2 complete. Distilled NN win rate vs elites: ${(distilledWinRate * 100).toFixed(1)}%`,
    });

    // ══════════════════════════════════════════
    //  PHASE 3: RL FINE-TUNING (PPO)
    // ══════════════════════════════════════════
    this.phase = 'ppo';
    onProgress?.({
      phase: 'ppo',
      phaseProgress: 0,
      message: 'Phase 3: Starting PPO self-play fine-tuning...',
    });

    const totalBatches = this.config.ppo.totalBatches || DEFAULT_PPO_CONFIG.totalBatches;
    const ppoTrainer = new PPOTrainer(this.distillationTrainer.model!, this.config.ppo);

    await ppoTrainer.train((stats: PPOStats) => {
      onProgress?.({
        phase: 'ppo',
        phaseProgress: stats.batch / totalBatches,
        message: `Batch ${stats.batch}/${totalBatches} | Reward: ${stats.avgReward.toFixed(3)} | Policy loss: ${stats.policyLoss.toFixed(4)} | Entropy: ${stats.entropy.toFixed(4)}`,
        ppoStats: stats,
      });
    });

    // Final validation
    const finalWinRate = await this.validateFinalModel();

    this.phase = 'done';
    onProgress?.({
      phase: 'done',
      phaseProgress: 1,
      message: `Pipeline complete! Final NN win rate vs genetic elites: ${(finalWinRate * 100).toFixed(1)}%`,
    });

    return this.distillationTrainer.model!;
  }

  /**
   * Validate the distilled model against the genetic elites.
   */
  private async validateDistilledModel(): Promise<number> {
    if (!this.distillationTrainer.model) return 0;

    const nnPolicy = new NNSimPolicy(this.distillationTrainer.model);
    let totalWins = 0;
    let totalGames = 0;

    for (const elite of this.elites) {
      const result = GameSimulator.playMatch(nnPolicy, elite, 20);
      totalWins += result.wins0;
      totalGames += 20;
    }

    return totalGames > 0 ? totalWins / totalGames : 0;
  }

  /**
   * Validate the final PPO-tuned model against the genetic elites.
   */
  private async validateFinalModel(): Promise<number> {
    return this.validateDistilledModel();
  }

  /** Get the current trained model */
  getModel(): tf.LayersModel | null {
    return this.distillationTrainer.model;
  }

  /** Get the elite policies from Phase 1 */
  getElites(): GeneticPolicy[] {
    return this.elites;
  }

  /** Get current phase */
  getPhase(): PipelinePhase {
    return this.phase;
  }

  /** Save the trained model */
  async saveModel(path: string): Promise<void> {
    await this.distillationTrainer.saveModel(path);
  }

  /** Load a previously trained model */
  async loadModel(path: string): Promise<void> {
    await this.distillationTrainer.loadModel(path);
  }
}
