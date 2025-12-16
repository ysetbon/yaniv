"""
Genetic Algorithm for Evolving Yaniv Neural Networks

Key features:
- Parallel evaluation using multiprocessing
- Evaluation against baselines + hall of fame + population samples
- Proper fitness calculation (not just win rate)
- Elitism + mutation selection
- Checkpoint saving and resumption
"""

import os
import json
import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

from .env import YanivEnv, play_game, Action
from .genome import Genome, Population
from .bots import RandomBot, GreedyBot, RuleBasedBot, BaseBot
from .model import TorchPolicy
from .encoding import StateActionEncoder


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Configuration for genetic algorithm."""
    # Population
    population_size: int = 128
    elite_count: int = 16

    # Network architecture
    input_dim: int = 281
    hidden_dim: int = 256
    hidden_dim2: int = 128

    # Evaluation
    games_per_opponent: int = 10
    baseline_weight: float = 0.4  # Fraction of games vs baselines
    hof_weight: float = 0.3  # Fraction vs hall of fame
    population_weight: float = 0.3  # Fraction vs population samples

    # Mutation
    mutation_rate: float = 0.02
    mutation_strength: float = 0.05
    mutation_rate_decay: float = 0.99  # Decay per generation
    mutation_strength_decay: float = 0.99

    # Crossover
    crossover_rate: float = 0.2

    # Hall of Fame
    hof_size: int = 10

    # Parallelization
    num_workers: int = 4
    batch_size: int = 16

    # Checkpointing
    checkpoint_dir: str = "ga_checkpoints"
    checkpoint_frequency: int = 5

    # Training
    num_generations: int = 100
    max_turns_per_game: int = 500
    random_seed: Optional[int] = None


class HallOfFame:
    """
    Hall of Fame - stores best genomes from previous generations.
    Prevents population from overfitting to current generation.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.members: List[Genome] = []

    def update(self, candidates: List[Genome]) -> None:
        """Update hall of fame with new candidates."""
        all_members = self.members + [g.copy() for g in candidates]
        # Sort by fitness and keep top
        all_members.sort(key=lambda g: g.fitness, reverse=True)
        self.members = all_members[:self.max_size]

    def get_members(self) -> List[Genome]:
        """Get hall of fame members."""
        return self.members

    def get_policy(self, idx: int, temperature: float = 0.0) -> TorchPolicy:
        """Get policy for a hall of fame member."""
        return self.members[idx].get_policy(temperature)

    def save(self, filepath: str) -> None:
        """Save hall of fame to file."""
        data = []
        for genome in self.members:
            weights = genome.get_weights()
            data.append({
                "genome_id": genome.genome_id,
                "fitness": genome.fitness,
                "generation": genome.generation,
                "weights": {k: v.tolist() for k, v in weights.items()}
            })
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: str, config: GAConfig) -> None:
        """Load hall of fame from file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.members = []
        for item in data:
            genome = Genome(
                input_dim=config.input_dim,
                hidden_dim=config.hidden_dim,
                hidden_dim2=config.hidden_dim2,
                genome_id=item["genome_id"]
            )
            weights = {k: np.array(v, dtype=np.float32) for k, v in item["weights"].items()}
            genome.set_weights(weights)
            genome.fitness = item["fitness"]
            genome.generation = item["generation"]
            self.members.append(genome)


def evaluate_genome_vs_bot(
    genome_weights: Dict[str, np.ndarray],
    bot_type: str,
    num_games: int,
    config_dict: Dict[str, Any],
    seed: int
) -> Tuple[int, int, float]:
    """
    Evaluate a genome against a baseline bot.

    Returns: (wins, losses, total_score_margin)
    """
    # Recreate genome from weights
    from .model import ActionValueNet, TorchPolicy
    from .encoding import StateActionEncoder

    model = ActionValueNet(
        input_dim=config_dict["input_dim"],
        hidden_dim=config_dict["hidden_dim"],
        hidden_dim2=config_dict["hidden_dim2"],
        use_torch=False  # Use numpy for multiprocessing
    )
    model.set_weights(genome_weights)
    policy = TorchPolicy(model, StateActionEncoder())

    # Create bot
    if bot_type == "random":
        bot = RandomBot(seed=seed)
    elif bot_type == "greedy":
        bot = GreedyBot(seed=seed)
    elif bot_type == "rule_based":
        bot = RuleBasedBot(seed=seed)
    else:
        raise ValueError(f"Unknown bot type: {bot_type}")

    wins = 0
    losses = 0
    total_margin = 0.0

    rng = random.Random(seed)

    for game_idx in range(num_games):
        game_seed = rng.randint(0, 2**31)
        env = YanivEnv(num_players=2, seed=game_seed, max_turns=config_dict["max_turns"])

        # Alternate who goes first
        if game_idx % 2 == 0:
            policies = [policy, bot]
            genome_idx = 0
        else:
            policies = [bot, policy]
            genome_idx = 1

        result = play_game(policies, env, seed=game_seed)

        if result["winner_idx"] == genome_idx:
            wins += 1
        else:
            losses += 1

        # Score margin (positive if genome did better)
        genome_score = result["scores"][f"player-{genome_idx}"]
        opponent_score = result["scores"][f"player-{1 - genome_idx}"]
        total_margin += opponent_score - genome_score

    return wins, losses, total_margin


def evaluate_genome_vs_genome(
    genome1_weights: Dict[str, np.ndarray],
    genome2_weights: Dict[str, np.ndarray],
    num_games: int,
    config_dict: Dict[str, Any],
    seed: int
) -> Tuple[int, int, float]:
    """
    Evaluate genome1 against genome2.

    Returns: (genome1_wins, genome1_losses, score_margin)
    """
    from .model import ActionValueNet, TorchPolicy
    from .encoding import StateActionEncoder

    model1 = ActionValueNet(
        input_dim=config_dict["input_dim"],
        hidden_dim=config_dict["hidden_dim"],
        hidden_dim2=config_dict["hidden_dim2"],
        use_torch=False
    )
    model1.set_weights(genome1_weights)
    policy1 = TorchPolicy(model1, StateActionEncoder())

    model2 = ActionValueNet(
        input_dim=config_dict["input_dim"],
        hidden_dim=config_dict["hidden_dim"],
        hidden_dim2=config_dict["hidden_dim2"],
        use_torch=False
    )
    model2.set_weights(genome2_weights)
    policy2 = TorchPolicy(model2, StateActionEncoder())

    wins = 0
    losses = 0
    total_margin = 0.0

    rng = random.Random(seed)

    for game_idx in range(num_games):
        game_seed = rng.randint(0, 2**31)
        env = YanivEnv(num_players=2, seed=game_seed, max_turns=config_dict["max_turns"])

        # Alternate who goes first
        if game_idx % 2 == 0:
            policies = [policy1, policy2]
            genome1_idx = 0
        else:
            policies = [policy2, policy1]
            genome1_idx = 1

        result = play_game(policies, env, seed=game_seed)

        if result["winner_idx"] == genome1_idx:
            wins += 1
        else:
            losses += 1

        genome1_score = result["scores"][f"player-{genome1_idx}"]
        genome2_score = result["scores"][f"player-{1 - genome1_idx}"]
        total_margin += genome2_score - genome1_score

    return wins, losses, total_margin


class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving Yaniv neural networks.
    """

    def __init__(self, config: GAConfig):
        self.config = config

        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        self.population = Population(
            size=config.population_size,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            hidden_dim2=config.hidden_dim2
        )

        self.hall_of_fame = HallOfFame(max_size=config.hof_size)

        # Current mutation parameters (may decay)
        self.current_mutation_rate = config.mutation_rate
        self.current_mutation_strength = config.mutation_strength

        # Statistics
        self.generation_times: List[float] = []
        self.evaluation_stats: List[Dict[str, Any]] = []

    def initialize(self) -> None:
        """Initialize population with random genomes."""
        logger.info(f"Initializing population of {self.config.population_size} genomes")
        self.population.initialize()

    def evaluate_population(self) -> None:
        """
        Evaluate all genomes in population.

        Each genome plays against:
        - Baseline bots (random, greedy, rule-based)
        - Hall of fame members
        - Random sample of other genomes
        """
        logger.info("Evaluating population...")
        start_time = time.time()

        # Reset fitness
        self.population.reset_fitness()

        # Prepare evaluation tasks
        config_dict = {
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "hidden_dim2": self.config.hidden_dim2,
            "max_turns": self.config.max_turns_per_game
        }

        # Calculate games per category
        total_games = self.config.games_per_opponent * 10
        baseline_games = int(total_games * self.config.baseline_weight)
        hof_games = int(total_games * self.config.hof_weight)
        pop_games = total_games - baseline_games - hof_games

        # Evaluate each genome
        for genome_idx, genome in enumerate(self.population.genomes):
            weights = genome.get_weights()
            seed_base = random.randint(0, 2**20)

            # Evaluate vs baselines
            for bot_type in ["random", "greedy", "rule_based"]:
                games = baseline_games // 3
                wins, losses, margin = evaluate_genome_vs_bot(
                    weights, bot_type, games, config_dict, seed_base
                )
                genome.wins += wins
                genome.losses += losses
                genome.total_score_margin += margin
                genome.games_played += games
                seed_base += 1000

            # Evaluate vs hall of fame
            if self.hall_of_fame.members:
                games_per_hof = max(1, hof_games // len(self.hall_of_fame.members))
                for hof_genome in self.hall_of_fame.members:
                    wins, losses, margin = evaluate_genome_vs_genome(
                        weights, hof_genome.get_weights(), games_per_hof,
                        config_dict, seed_base
                    )
                    genome.wins += wins
                    genome.losses += losses
                    genome.total_score_margin += margin
                    genome.games_played += games_per_hof
                    seed_base += 1000

            # Evaluate vs random population samples
            other_genomes = [
                g for i, g in enumerate(self.population.genomes) if i != genome_idx
            ]
            sample_size = min(5, len(other_genomes))
            sampled = random.sample(other_genomes, sample_size) if other_genomes else []
            games_per_sample = max(1, pop_games // max(1, sample_size))

            for other in sampled:
                wins, losses, margin = evaluate_genome_vs_genome(
                    weights, other.get_weights(), games_per_sample,
                    config_dict, seed_base
                )
                genome.wins += wins
                genome.losses += losses
                genome.total_score_margin += margin
                genome.games_played += games_per_sample
                seed_base += 1000

            # Calculate fitness
            genome.calculate_fitness()

            if (genome_idx + 1) % 10 == 0:
                logger.info(f"  Evaluated {genome_idx + 1}/{len(self.population.genomes)} genomes")

        eval_time = time.time() - start_time
        logger.info(f"Evaluation completed in {eval_time:.1f}s")

        # Record stats
        best = self.population.get_best()
        avg_fitness = self.population.get_average_fitness()
        self.evaluation_stats.append({
            "generation": self.population.generation,
            "best_fitness": best.fitness,
            "avg_fitness": avg_fitness,
            "best_win_rate": best.wins / max(1, best.games_played),
            "eval_time": eval_time
        })

    def evaluate_population_parallel(self) -> None:
        """
        Parallel evaluation of population using multiprocessing.
        """
        logger.info(f"Evaluating population (parallel, {self.config.num_workers} workers)...")
        start_time = time.time()

        self.population.reset_fitness()

        config_dict = {
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "hidden_dim2": self.config.hidden_dim2,
            "max_turns": self.config.max_turns_per_game
        }

        # Calculate games per category
        total_games = self.config.games_per_opponent * 10
        baseline_games = int(total_games * self.config.baseline_weight)

        # Prepare tasks
        tasks = []
        for genome_idx, genome in enumerate(self.population.genomes):
            weights = genome.get_weights()
            seed_base = random.randint(0, 2**20) + genome_idx * 10000

            for bot_type in ["random", "greedy", "rule_based"]:
                games = baseline_games // 3
                tasks.append((genome_idx, "bot", weights, bot_type, games, seed_base))
                seed_base += 1000

        # Execute in parallel
        results_by_genome: Dict[int, List[Tuple[int, int, float]]] = {
            i: [] for i in range(len(self.population.genomes))
        }

        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {}
            for task in tasks:
                genome_idx, task_type, weights, bot_type, games, seed = task
                future = executor.submit(
                    evaluate_genome_vs_bot,
                    weights, bot_type, games, config_dict, seed
                )
                futures[future] = genome_idx

            for future in as_completed(futures):
                genome_idx = futures[future]
                try:
                    wins, losses, margin = future.result()
                    results_by_genome[genome_idx].append((wins, losses, margin))
                except Exception as e:
                    logger.error(f"Error evaluating genome {genome_idx}: {e}")

        # Aggregate results
        for genome_idx, results in results_by_genome.items():
            genome = self.population.genomes[genome_idx]
            for wins, losses, margin in results:
                genome.wins += wins
                genome.losses += losses
                genome.total_score_margin += margin
                genome.games_played += wins + losses
            genome.calculate_fitness()

        eval_time = time.time() - start_time
        logger.info(f"Parallel evaluation completed in {eval_time:.1f}s")

        # Record stats
        best = self.population.get_best()
        avg_fitness = self.population.get_average_fitness()
        self.evaluation_stats.append({
            "generation": self.population.generation,
            "best_fitness": best.fitness,
            "avg_fitness": avg_fitness,
            "best_win_rate": best.wins / max(1, best.games_played),
            "eval_time": eval_time
        })

    def select_and_reproduce(self) -> None:
        """
        Selection and reproduction to create next generation.
        """
        logger.info("Selection and reproduction...")

        # Get elite genomes
        elites = self.population.get_elite(self.config.elite_count)
        logger.info(f"  Top elite fitness: {elites[0].fitness:.3f}")

        # Update hall of fame
        self.hall_of_fame.update(elites[:3])

        # Create next generation
        next_gen: List[Genome] = []

        # Keep elites unchanged
        for elite in elites:
            elite_copy = elite.copy()
            elite_copy.generation = self.population.generation + 1
            next_gen.append(elite_copy)

        # Fill remaining with mutations and crossovers
        while len(next_gen) < self.config.population_size:
            if random.random() < self.config.crossover_rate and len(elites) >= 2:
                # Crossover
                parent1, parent2 = random.sample(elites, 2)
                child = parent1.crossover(parent2)
                child = child.mutate(
                    self.current_mutation_rate,
                    self.current_mutation_strength
                )
            else:
                # Mutation only
                parent = random.choice(elites)
                child = parent.mutate(
                    self.current_mutation_rate,
                    self.current_mutation_strength
                )

            child.generation = self.population.generation + 1
            next_gen.append(child)

        self.population.genomes = next_gen
        self.population.generation += 1

        # Decay mutation parameters
        self.current_mutation_rate *= self.config.mutation_rate_decay
        self.current_mutation_strength *= self.config.mutation_strength_decay

        logger.info(f"  Next generation: {len(next_gen)} genomes")
        logger.info(f"  Mutation rate: {self.current_mutation_rate:.4f}")

    def save_checkpoint(self) -> None:
        """Save checkpoint of current state."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        gen = self.population.generation
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, f"gen_{gen:04d}"
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save best genome
        best = self.population.get_best()
        best.save(os.path.join(checkpoint_path, "best_genome.json"))

        # Save hall of fame
        self.hall_of_fame.save(os.path.join(checkpoint_path, "hall_of_fame.json"))

        # Save statistics
        stats = {
            "generation": gen,
            "best_fitness": best.fitness,
            "avg_fitness": self.population.get_average_fitness(),
            "mutation_rate": self.current_mutation_rate,
            "mutation_strength": self.current_mutation_strength,
            "evaluation_stats": self.evaluation_stats
        }
        with open(os.path.join(checkpoint_path, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load best genome to reconstruct population
        best_path = os.path.join(checkpoint_path, "best_genome.json")
        if os.path.exists(best_path):
            best = Genome.load(best_path)
            # Reinitialize population with mutations of best
            self.population.genomes = [best.copy()]
            for _ in range(self.config.population_size - 1):
                mutated = best.mutate(self.current_mutation_rate, self.current_mutation_strength)
                self.population.genomes.append(mutated)

        # Load hall of fame
        hof_path = os.path.join(checkpoint_path, "hall_of_fame.json")
        if os.path.exists(hof_path):
            self.hall_of_fame.load(hof_path, self.config)

        # Load stats
        stats_path = os.path.join(checkpoint_path, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            self.population.generation = stats["generation"]
            self.current_mutation_rate = stats.get("mutation_rate", self.config.mutation_rate)
            self.current_mutation_strength = stats.get("mutation_strength", self.config.mutation_strength)
            self.evaluation_stats = stats.get("evaluation_stats", [])

    def run(self, parallel: bool = False) -> Genome:
        """
        Run the genetic algorithm.

        Args:
            parallel: Use parallel evaluation

        Returns:
            Best genome found
        """
        logger.info("=" * 60)
        logger.info("Starting Genetic Algorithm")
        logger.info(f"Population: {self.config.population_size}")
        logger.info(f"Generations: {self.config.num_generations}")
        logger.info(f"Elite count: {self.config.elite_count}")
        logger.info("=" * 60)

        if not self.population.genomes:
            self.initialize()

        start_gen = self.population.generation

        for gen in range(start_gen, start_gen + self.config.num_generations):
            gen_start = time.time()
            logger.info(f"\n=== Generation {gen} ===")

            # Evaluate
            if parallel:
                self.evaluate_population_parallel()
            else:
                self.evaluate_population()

            # Log stats
            best = self.population.get_best()
            avg = self.population.get_average_fitness()
            logger.info(f"Best fitness: {best.fitness:.4f}")
            logger.info(f"Avg fitness: {avg:.4f}")
            logger.info(f"Best win rate: {best.wins}/{best.games_played}")

            # Record generation stats
            self.population.record_generation_stats()

            # Checkpoint
            if (gen + 1) % self.config.checkpoint_frequency == 0:
                self.save_checkpoint()

            # Select and reproduce (skip on last generation)
            if gen < start_gen + self.config.num_generations - 1:
                self.select_and_reproduce()

            gen_time = time.time() - gen_start
            self.generation_times.append(gen_time)
            logger.info(f"Generation time: {gen_time:.1f}s")

        # Final save
        self.save_checkpoint()

        # Save final best model
        best = self.population.get_best()
        final_path = os.path.join(self.config.checkpoint_dir, "best_final.json")
        best.save(final_path)
        logger.info(f"Final best genome saved to {final_path}")

        return best


def train(
    generations: int = 100,
    population_size: int = 128,
    elite_count: int = 16,
    checkpoint_dir: str = "ga_checkpoints",
    num_workers: int = 4,
    parallel: bool = True,
    resume_from: Optional[str] = None
) -> Genome:
    """
    Convenience function to train with default settings.
    """
    config = GAConfig(
        num_generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        checkpoint_dir=checkpoint_dir,
        num_workers=num_workers
    )

    ga = GeneticAlgorithm(config)

    if resume_from:
        ga.load_checkpoint(resume_from)
    else:
        ga.initialize()

    return ga.run(parallel=parallel)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Yaniv AI with Genetic Algorithm")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--population", type=int, default=64, help="Population size")
    parser.add_argument("--elite", type=int, default=8, help="Elite count")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--checkpoint-dir", type=str, default="ga_checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--sequential", action="store_true", help="Use sequential evaluation")

    args = parser.parse_args()

    best = train(
        generations=args.generations,
        population_size=args.population,
        elite_count=args.elite,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.workers,
        parallel=not args.sequential,
        resume_from=args.resume
    )

    print(f"\nTraining complete!")
    print(f"Best genome: {best}")
