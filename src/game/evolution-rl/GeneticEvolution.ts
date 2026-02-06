/**
 * Phase 1: Genetic Evolution
 *
 * Evolves a population of GeneticPolicy instances through tournament play.
 *
 * Pipeline:
 *  1. Generate N random policies
 *  2. Evaluate fitness via round-robin tournament
 *  3. Select top performers (elitism + tournament selection)
 *  4. Create next generation via crossover + mutation
 *  5. Repeat for G generations
 *  6. Return top K elite policies
 */

import { GeneticPolicy } from './GeneticPolicy';
import { GameSimulator } from './GameSimulator';

export interface EvolutionConfig {
  populationSize: number;    // Number of policies per generation (default: 30)
  generations: number;       // Number of evolution generations (default: 50)
  gamesPerMatchup: number;   // Games per matchup for fitness eval (default: 6)
  eliteCount: number;        // Number of elites to preserve (default: 5)
  mutationRate: number;      // Per-gene mutation probability (default: 0.2)
  mutationStrength: number;  // Mutation magnitude (default: 0.5)
  tournamentSize: number;    // Tournament selection size (default: 4)
  crossoverRate: number;     // Probability of crossover vs. mutation-only (default: 0.7)
  matchupsPerPolicy: number; // Number of opponents to play against per eval (default: 8)
}

export const DEFAULT_EVOLUTION_CONFIG: EvolutionConfig = {
  populationSize: 30,
  generations: 50,
  gamesPerMatchup: 6,
  eliteCount: 5,
  mutationRate: 0.2,
  mutationStrength: 0.5,
  tournamentSize: 4,
  crossoverRate: 0.7,
  matchupsPerPolicy: 8,
};

export interface GenerationStats {
  generation: number;
  bestFitness: number;
  avgFitness: number;
  worstFitness: number;
  eliteFitnesses: number[];
}

export type EvolutionProgressCallback = (stats: GenerationStats) => void;

export class GeneticEvolution {
  private config: EvolutionConfig;
  private population: GeneticPolicy[] = [];

  constructor(config?: Partial<EvolutionConfig>) {
    this.config = { ...DEFAULT_EVOLUTION_CONFIG, ...config };
  }

  /**
   * Run the full evolution process and return the top elite policies.
   */
  evolve(onProgress?: EvolutionProgressCallback): GeneticPolicy[] {
    // Step 1: Initialize random population
    this.population = Array.from(
      { length: this.config.populationSize },
      () => new GeneticPolicy(),
    );

    for (let gen = 0; gen < this.config.generations; gen++) {
      // Step 2: Evaluate fitness
      this.evaluateFitness();

      // Sort by fitness (descending)
      this.population.sort((a, b) => b.fitness - a.fitness);

      // Report progress
      const stats: GenerationStats = {
        generation: gen + 1,
        bestFitness: this.population[0].fitness,
        avgFitness: this.population.reduce((s, p) => s + p.fitness, 0) / this.population.length,
        worstFitness: this.population[this.population.length - 1].fitness,
        eliteFitnesses: this.population.slice(0, this.config.eliteCount).map(p => p.fitness),
      };
      onProgress?.(stats);

      // Don't evolve on the last generation
      if (gen === this.config.generations - 1) break;

      // Step 3-4: Create next generation
      this.population = this.createNextGeneration();
    }

    // Return top elites
    this.population.sort((a, b) => b.fitness - a.fitness);
    return this.population.slice(0, this.config.eliteCount);
  }

  /**
   * Evaluate fitness of all policies via tournament play.
   * Each policy plays against a random subset of opponents.
   */
  private evaluateFitness(): void {
    // Reset fitness
    for (const policy of this.population) {
      policy.fitness = 0;
    }

    for (let i = 0; i < this.population.length; i++) {
      const policy = this.population[i];
      let totalGames = 0;
      let totalWins = 0;

      // Select random opponents
      const opponentIndices = this.selectRandomOpponents(i, this.config.matchupsPerPolicy);

      for (const oppIdx of opponentIndices) {
        const opponent = this.population[oppIdx];
        const result = GameSimulator.playMatch(
          policy, opponent, this.config.gamesPerMatchup,
        );
        totalWins += result.wins0;
        totalGames += this.config.gamesPerMatchup;
      }

      policy.fitness = totalGames > 0 ? totalWins / totalGames : 0;
    }
  }

  /** Select random opponent indices, excluding self */
  private selectRandomOpponents(selfIdx: number, count: number): number[] {
    const indices: number[] = [];
    const available = Array.from(
      { length: this.population.length },
      (_, i) => i,
    ).filter(i => i !== selfIdx);

    const n = Math.min(count, available.length);
    for (let i = 0; i < n; i++) {
      const randIdx = Math.floor(Math.random() * available.length);
      indices.push(available[randIdx]);
      available.splice(randIdx, 1);
    }
    return indices;
  }

  /**
   * Create next generation using elitism, tournament selection, crossover, mutation.
   */
  private createNextGeneration(): GeneticPolicy[] {
    const nextGen: GeneticPolicy[] = [];

    // Elitism: carry forward top performers unchanged
    const elites = this.population.slice(0, this.config.eliteCount);
    for (const elite of elites) {
      const clone = GeneticPolicy.fromArray(elite.toArray());
      clone.fitness = elite.fitness;
      nextGen.push(clone);
    }

    // Fill remaining slots
    while (nextGen.length < this.config.populationSize) {
      if (Math.random() < this.config.crossoverRate) {
        // Crossover: select two parents via tournament, create child
        const parent1 = this.tournamentSelect();
        const parent2 = this.tournamentSelect();
        const child = GeneticPolicy.crossover(parent1, parent2);
        const mutated = GeneticPolicy.mutate(
          child, this.config.mutationRate, this.config.mutationStrength,
        );
        nextGen.push(mutated);
      } else {
        // Mutation only: select one parent, mutate it
        const parent = this.tournamentSelect();
        const mutated = GeneticPolicy.mutate(
          parent, this.config.mutationRate * 1.5, this.config.mutationStrength * 1.2,
        );
        nextGen.push(mutated);
      }
    }

    return nextGen;
  }

  /** Tournament selection: pick best from a random subset */
  private tournamentSelect(): GeneticPolicy {
    let best: GeneticPolicy | null = null;

    for (let i = 0; i < this.config.tournamentSize; i++) {
      const idx = Math.floor(Math.random() * this.population.length);
      const candidate = this.population[idx];
      if (!best || candidate.fitness > best.fitness) {
        best = candidate;
      }
    }

    return best!;
  }
}
