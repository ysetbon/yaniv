"""
Genome class for Genetic Algorithm

A Genome wraps a neural network with:
- Weight storage (state_dict)
- Fitness tracking
- Mutation operations
- Crossover support
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import json
import copy

from .model import ActionValueNet, TorchPolicy
from .encoding import StateActionEncoder


class Genome:
    """
    Genome for genetic algorithm optimization.

    Each genome contains:
    - Neural network weights
    - Fitness score and statistics
    - Generation metadata
    """

    def __init__(
        self,
        model: Optional[ActionValueNet] = None,
        input_dim: int = 281,
        hidden_dim: int = 256,
        hidden_dim2: int = 128,
        genome_id: Optional[str] = None
    ):
        self.genome_id = genome_id or self._generate_id()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        # Initialize or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = ActionValueNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                hidden_dim2=hidden_dim2,
                use_torch=True
            )

        # Fitness tracking
        self.fitness: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.games_played: int = 0
        self.total_score_margin: float = 0.0

        # Metadata
        self.generation: int = 0
        self.parent_ids: List[str] = []

    def _generate_id(self) -> str:
        """Generate unique genome ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights."""
        return self.model.get_weights()

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set model weights."""
        self.model.set_weights(weights)

    def mutate(
        self,
        mutation_rate: float = 0.02,
        mutation_strength: float = 0.05
    ) -> "Genome":
        """
        Create mutated copy of this genome.

        Args:
            mutation_rate: Probability each weight is mutated
            mutation_strength: Standard deviation of Gaussian noise

        Returns:
            New mutated genome
        """
        new_genome = self.copy()
        new_genome.genome_id = self._generate_id()
        new_genome.parent_ids = [self.genome_id]
        new_genome.reset_fitness()

        weights = new_genome.get_weights()
        mutated_weights = {}

        for key, value in weights.items():
            # Create mutation mask
            mask = np.random.random(value.shape) < mutation_rate
            # Gaussian noise
            noise = np.random.randn(*value.shape).astype(np.float32) * mutation_strength
            # Apply mutation only where mask is True
            mutated_weights[key] = value + mask * noise

        new_genome.set_weights(mutated_weights)
        return new_genome

    def crossover(self, other: "Genome") -> "Genome":
        """
        Create child genome via uniform crossover.

        Args:
            other: Other parent genome

        Returns:
            New child genome with mixed weights
        """
        child = Genome(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            hidden_dim2=self.hidden_dim2
        )
        child.parent_ids = [self.genome_id, other.genome_id]
        child.generation = max(self.generation, other.generation) + 1

        weights1 = self.get_weights()
        weights2 = other.get_weights()
        child_weights = {}

        for key in weights1.keys():
            # Uniform crossover: randomly pick from each parent
            mask = np.random.random(weights1[key].shape) < 0.5
            child_weights[key] = np.where(mask, weights1[key], weights2[key])

        child.set_weights(child_weights)
        return child

    def copy(self) -> "Genome":
        """Create deep copy of genome."""
        new_genome = Genome(
            model=self.model.copy(),
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            hidden_dim2=self.hidden_dim2,
            genome_id=self.genome_id
        )
        new_genome.fitness = self.fitness
        new_genome.wins = self.wins
        new_genome.losses = self.losses
        new_genome.games_played = self.games_played
        new_genome.total_score_margin = self.total_score_margin
        new_genome.generation = self.generation
        new_genome.parent_ids = list(self.parent_ids)
        return new_genome

    def reset_fitness(self) -> None:
        """Reset fitness tracking for new evaluation."""
        self.fitness = 0.0
        self.wins = 0
        self.losses = 0
        self.games_played = 0
        self.total_score_margin = 0.0

    def record_game(
        self,
        won: bool,
        score_margin: float = 0.0
    ) -> None:
        """Record game result."""
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        self.total_score_margin += score_margin

    def calculate_fitness(self) -> float:
        """
        Calculate fitness from game statistics.

        Fitness = 0.7 * win_rate + 0.3 * normalized_score_margin
        """
        if self.games_played == 0:
            self.fitness = 0.0
            return self.fitness

        win_rate = self.wins / self.games_played

        # Normalize score margin to [0, 1] range
        # Assuming score margin typically in [-100, 100]
        avg_margin = self.total_score_margin / self.games_played
        normalized_margin = (avg_margin + 100) / 200  # Map to [0, 1]
        normalized_margin = np.clip(normalized_margin, 0, 1)

        self.fitness = 0.7 * win_rate + 0.3 * normalized_margin
        return self.fitness

    def get_policy(self, temperature: float = 0.0) -> TorchPolicy:
        """Get policy wrapper for this genome."""
        return TorchPolicy(
            model=self.model,
            encoder=StateActionEncoder(),
            temperature=temperature
        )

    def save(self, filepath: str) -> None:
        """Save genome to JSON file."""
        data = {
            "genome_id": self.genome_id,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_dim2": self.hidden_dim2,
            "fitness": self.fitness,
            "wins": self.wins,
            "losses": self.losses,
            "games_played": self.games_played,
            "total_score_margin": self.total_score_margin,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "weights": {k: v.tolist() for k, v in self.get_weights().items()}
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Genome":
        """Load genome from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        genome = cls(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            hidden_dim2=data["hidden_dim2"],
            genome_id=data["genome_id"]
        )

        weights = {k: np.array(v, dtype=np.float32) for k, v in data["weights"].items()}
        genome.set_weights(weights)

        genome.fitness = data["fitness"]
        genome.wins = data["wins"]
        genome.losses = data["losses"]
        genome.games_played = data["games_played"]
        genome.total_score_margin = data["total_score_margin"]
        genome.generation = data["generation"]
        genome.parent_ids = data["parent_ids"]

        return genome

    def __repr__(self) -> str:
        return (
            f"Genome({self.genome_id}, "
            f"fitness={self.fitness:.3f}, "
            f"wins={self.wins}/{self.games_played}, "
            f"gen={self.generation})"
        )


class Population:
    """
    Population of genomes for GA.
    """

    def __init__(
        self,
        size: int = 128,
        input_dim: int = 281,
        hidden_dim: int = 256,
        hidden_dim2: int = 128
    ):
        self.size = size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        self.genomes: List[Genome] = []
        self.generation: int = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

    def initialize(self) -> None:
        """Initialize random population."""
        self.genomes = [
            Genome(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                hidden_dim2=self.hidden_dim2
            )
            for _ in range(self.size)
        ]

    def get_sorted(self) -> List[Genome]:
        """Get genomes sorted by fitness (best first)."""
        return sorted(self.genomes, key=lambda g: g.fitness, reverse=True)

    def get_elite(self, count: int) -> List[Genome]:
        """Get top N genomes."""
        return self.get_sorted()[:count]

    def get_best(self) -> Genome:
        """Get best genome."""
        return max(self.genomes, key=lambda g: g.fitness)

    def get_average_fitness(self) -> float:
        """Get average fitness of population."""
        if not self.genomes:
            return 0.0
        return sum(g.fitness for g in self.genomes) / len(self.genomes)

    def reset_fitness(self) -> None:
        """Reset fitness for all genomes."""
        for genome in self.genomes:
            genome.reset_fitness()

    def record_generation_stats(self) -> None:
        """Record generation statistics."""
        self.best_fitness_history.append(self.get_best().fitness)
        self.avg_fitness_history.append(self.get_average_fitness())

    def save(self, dirpath: str) -> None:
        """Save population to directory."""
        import os
        os.makedirs(dirpath, exist_ok=True)

        # Save metadata
        meta = {
            "size": self.size,
            "generation": self.generation,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_dim2": self.hidden_dim2,
            "best_fitness_history": self.best_fitness_history,
            "avg_fitness_history": self.avg_fitness_history
        }
        with open(os.path.join(dirpath, "population_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Save genomes
        for i, genome in enumerate(self.genomes):
            genome.save(os.path.join(dirpath, f"genome_{i:04d}.json"))

    @classmethod
    def load(cls, dirpath: str) -> "Population":
        """Load population from directory."""
        import os

        with open(os.path.join(dirpath, "population_meta.json"), "r") as f:
            meta = json.load(f)

        pop = cls(
            size=meta["size"],
            input_dim=meta["input_dim"],
            hidden_dim=meta["hidden_dim"],
            hidden_dim2=meta["hidden_dim2"]
        )
        pop.generation = meta["generation"]
        pop.best_fitness_history = meta.get("best_fitness_history", [])
        pop.avg_fitness_history = meta.get("avg_fitness_history", [])

        # Load genomes
        pop.genomes = []
        for i in range(pop.size):
            filepath = os.path.join(dirpath, f"genome_{i:04d}.json")
            if os.path.exists(filepath):
                pop.genomes.append(Genome.load(filepath))

        return pop
