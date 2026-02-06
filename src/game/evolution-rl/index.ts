/**
 * Evolution → Distillation → RL Pipeline
 *
 * Three-phase AI training system for Yaniv:
 *
 *  Phase 1 (Genetic Evolution):
 *    GeneticPolicy + GeneticEvolution → Elite heuristic policies
 *
 *  Phase 2 (Distillation):
 *    DistillationTrainer → Neural network that imitates elites
 *
 *  Phase 3 (RL Fine-tuning):
 *    PPOTrainer → NN surpasses genetic ceiling via self-play
 *
 *  Result: EvolutionRLAI → Playable AI for the game
 */

export { encodeState, STATE_SIZE, cardToIndex, indexToCard, enumerateValidDiscards } from './StateEncoder';
export type { GameSnapshot } from './StateEncoder';
export { GeneticPolicy, GENOME_SIZE } from './GeneticPolicy';
export type { GeneticPolicyGenome } from './GeneticPolicy';
export { GameSimulator } from './GameSimulator';
export type { SimPolicy, GameResult, TrajectoryStep, RecordedAction } from './GameSimulator';
export { GeneticEvolution, DEFAULT_EVOLUTION_CONFIG } from './GeneticEvolution';
export type { EvolutionConfig, GenerationStats } from './GeneticEvolution';
export { DistillationTrainer, DEFAULT_DISTILLATION_CONFIG } from './DistillationTrainer';
export type { DistillationConfig, DistillationStats } from './DistillationTrainer';
export { PPOTrainer, DEFAULT_PPO_CONFIG, NNSimPolicy } from './PPOTrainer';
export type { PPOConfig, PPOStats } from './PPOTrainer';
export { EvolutionRLPipeline, DEFAULT_PIPELINE_CONFIG } from './EvolutionRLPipeline';
export type { PipelineConfig, PipelineProgress, PipelinePhase } from './EvolutionRLPipeline';
export { EvolutionRLAI } from './EvolutionRLAI';
