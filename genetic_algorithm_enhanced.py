import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
# Removed multiprocessing for compatibility
import pickle
import json
from yaniv_neural_network_enhanced import EnhancedYanivNN, AdvancedFeatureExtractor, HallOfFame, EnsembleAI
from yaniv_game_enhanced_wrapper import play_simple_match
import time


class AdaptiveGeneticAlgorithm:
    """Enhanced genetic algorithm with adaptive mutation rates and advanced operators"""
    
    def __init__(self, population_size: int = 100, elite_size: int = 10,
                 mutation_rate_initial: float = 0.1, crossover_rate: float = 0.7,
                 tournament_size: int = 5):
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Adaptive mutation rates for each individual
        self.mutation_rates = np.ones(population_size) * mutation_rate_initial
        self.fitness_history = []
        self.stagnation_counter = 0
        
        # Hall of Fame to preserve best networks
        self.hall_of_fame = HallOfFame(size=20)
        
        # Initialize population with enhanced networks
        self.population = [EnhancedYanivNN() for _ in range(population_size)]
        self.fitness_scores = np.zeros(population_size)
        
        # Feature extractor
        self.feature_extractor = AdvancedFeatureExtractor()
        
        # Performance tracking
        self.generation_stats = []
        
    def evaluate_fitness(self, individual_idx: int, num_games: int = 10, 
                        include_hall_of_fame: bool = True) -> float:
        """Evaluate fitness against diverse opponents"""
        network = self.population[individual_idx]
        total_wins = 0
        games_played = 0
        
        # Play against other population members
        opponents_idx = np.random.choice(
            [i for i in range(self.population_size) if i != individual_idx],
            size=min(5, self.population_size - 1),
            replace=False
        )
        
        for opp_idx in opponents_idx:
            wins = self._play_matches(network, self.population[opp_idx], num_games=2)
            total_wins += wins
            games_played += 2
        
        # Play against Hall of Fame champions
        if include_hall_of_fame and len(self.hall_of_fame.champions) > 0:
            hof_opponents = self.hall_of_fame.get_opponents(n=3)
            for hof_opponent in hof_opponents:
                wins = self._play_matches(network, hof_opponent, num_games=2)
                total_wins += wins
                games_played += 2
        
        # Calculate fitness with bonus for beating champions
        base_fitness = total_wins / games_played if games_played > 0 else 0
        
        # Bonus for consistency
        if games_played >= 10:
            consistency_bonus = 0.1 * (1 - np.std([total_wins / games_played]))
            base_fitness += consistency_bonus
        
        return base_fitness
    
    def _play_matches(self, network1: EnhancedYanivNN, network2: EnhancedYanivNN, 
                     num_games: int = 2) -> int:
        """Play matches between two networks"""
        wins = 0
        
        for game_idx in range(num_games):
            # Alternate who goes first
            if game_idx % 2 == 0:
                players = [network1, network2]
                player1_idx = 0
            else:
                players = [network2, network1]
                player1_idx = 1
            
            game_wins = play_simple_match(network1, network2)
            wins += game_wins
        
        return wins
    
    def adapt_mutation_rate(self, individual_idx: int):
        """Adapt mutation rate based on fitness improvement"""
        if len(self.fitness_history) < 2:
            return
        
        current_fitness = self.fitness_scores[individual_idx]
        
        # Check improvement over last few generations
        recent_history = self.fitness_history[-5:]
        avg_recent = np.mean([gen['best'] for gen in recent_history])
        
        if current_fitness > avg_recent * 1.05:  # 5% improvement
            # Reduce mutation for good performers
            self.mutation_rates[individual_idx] *= 0.9
        elif current_fitness < avg_recent * 0.95:  # 5% worse
            # Increase mutation for poor performers
            self.mutation_rates[individual_idx] *= 1.1
        
        # Bounds
        self.mutation_rates[individual_idx] = np.clip(
            self.mutation_rates[individual_idx], 0.01, 0.5
        )
    
    def mutate(self, individual_idx: int, strength_multiplier: float = 1.0):
        """Mutate individual with adaptive rate"""
        network = self.population[individual_idx]
        mutation_rate = self.mutation_rates[individual_idx] * strength_multiplier
        
        # Get flat weights
        weights = network.get_weights_flat()
        
        # Apply mutations
        mask = np.random.random(weights.shape) < mutation_rate
        mutations = np.random.randn(weights.shape[0]) * 0.1
        weights[mask] += mutations[mask]
        
        # Occasionally do larger mutations
        if np.random.random() < 0.1:  # 10% chance
            large_mutation_idx = np.random.choice(len(weights), size=int(len(weights) * 0.01))
            weights[large_mutation_idx] += np.random.randn(len(large_mutation_idx)) * 0.5
        
        # Set weights back
        network.set_weights_from_flat(weights)
    
    def crossover(self, parent1_idx: int, parent2_idx: int) -> EnhancedYanivNN:
        """Advanced crossover operation"""
        parent1 = self.population[parent1_idx]
        parent2 = self.population[parent2_idx]
        
        child = EnhancedYanivNN()
        
        # Get parent weights
        weights1 = parent1.get_weights_flat()
        weights2 = parent2.get_weights_flat()
        
        # Use different crossover strategies
        strategy = np.random.choice(['uniform', 'block', 'arithmetic'])
        
        if strategy == 'uniform':
            # Uniform crossover
            mask = np.random.random(weights1.shape) < 0.5
            child_weights = np.where(mask, weights1, weights2)
            
        elif strategy == 'block':
            # Block crossover - swap large chunks
            num_blocks = 10
            block_size = len(weights1) // num_blocks
            child_weights = weights1.copy()
            
            for i in range(num_blocks):
                if np.random.random() < 0.5:
                    start = i * block_size
                    end = min((i + 1) * block_size, len(weights1))
                    child_weights[start:end] = weights2[start:end]
                    
        else:  # arithmetic
            # Arithmetic crossover - weighted average
            alpha = np.random.uniform(0.3, 0.7)
            child_weights = alpha * weights1 + (1 - alpha) * weights2
        
        child.set_weights_from_flat(child_weights)
        return child
    
    def tournament_selection(self, tournament_size: Optional[int] = None) -> int:
        """Select individual through tournament"""
        if tournament_size is None:
            tournament_size = self.tournament_size
            
        candidates = np.random.choice(self.population_size, size=tournament_size, replace=False)
        fitnesses = self.fitness_scores[candidates]
        winner_idx = candidates[np.argmax(fitnesses)]
        
        return winner_idx
    
    def evolve_generation(self):
        """Evolve one generation with advanced techniques"""
        print(f"Evaluating generation {len(self.fitness_history) + 1}...")
        
        # Sequential fitness evaluation (more reliable)
        print(f"Evaluating {self.population_size} individuals...")
        for i in range(self.population_size):
            if i % 5 == 0:
                print(f"  Progress: {i}/{self.population_size}")
            self.fitness_scores[i] = self.evaluate_fitness(i)
        
        # Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Update Hall of Fame
        best_idx = sorted_indices[0]
        best_fitness = self.fitness_scores[best_idx]
        self.hall_of_fame.update(
            self.population[best_idx],
            best_fitness,
            len(self.fitness_history) + 1,
            {'mutation_rate': self.mutation_rates[best_idx]}
        )
        
        # Track statistics
        stats = {
            'best': best_fitness,
            'mean': np.mean(self.fitness_scores),
            'std': np.std(self.fitness_scores),
            'min': np.min(self.fitness_scores),
            'mutation_rates': {
                'mean': np.mean(self.mutation_rates),
                'std': np.std(self.mutation_rates)
            }
        }
        self.generation_stats.append(stats)
        
        print(f"Generation {len(self.fitness_history) + 1} - "
              f"Best: {stats['best']:.3f}, Mean: {stats['mean']:.3f}, "
              f"Std: {stats['std']:.3f}")
        
        # Check for stagnation
        if len(self.fitness_history) > 0:
            if best_fitness <= self.fitness_history[-1]['best'] * 1.01:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        self.fitness_history.append(stats)
        
        # Create new population
        new_population = []
        
        # Elitism - keep best individuals
        for i in range(self.elite_size):
            elite = EnhancedYanivNN()
            elite_weights = self.population[sorted_indices[i]].get_weights_flat()
            elite.set_weights_from_flat(elite_weights.copy())
            new_population.append(elite)
        
        # Fill rest of population
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate:
                # Crossover
                parent1_idx = self.tournament_selection()
                parent2_idx = self.tournament_selection()
                child = self.crossover(parent1_idx, parent2_idx)
                new_population.append(child)
            else:
                # Clone and mutate
                parent_idx = self.tournament_selection()
                child = EnhancedYanivNN()
                parent_weights = self.population[parent_idx].get_weights_flat()
                child.set_weights_from_flat(parent_weights.copy())
                new_population.append(child)
        
        # Update population
        self.population = new_population[:self.population_size]
        
        # Apply mutations (skip elites)
        strength_multiplier = 1.0
        if self.stagnation_counter > 5:
            strength_multiplier = 1.5  # Increase mutation strength
            print("Increasing mutation strength due to stagnation")
        
        for i in range(self.elite_size, self.population_size):
            self.adapt_mutation_rate(i)
            self.mutate(i, strength_multiplier)
        
        # Reset fitness scores
        self.fitness_scores = np.zeros(self.population_size)
    
    def create_ensemble(self, top_n: int = 5) -> EnsembleAI:
        """Create ensemble from best networks"""
        # Get top networks from current population
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        top_networks = [self.population[idx] for idx in sorted_indices[:top_n]]
        
        # Add some from Hall of Fame for diversity
        if len(self.hall_of_fame.champions) > 0:
            hof_networks = self.hall_of_fame.get_opponents(n=min(3, top_n // 2))
            top_networks.extend(hof_networks)
        
        # Create ensemble with equal weights initially
        ensemble = EnsembleAI(top_networks[:top_n])
        
        return ensemble
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'generation': len(self.fitness_history),
            'population_size': self.population_size,
            'fitness_scores': self.fitness_scores.tolist(),
            'mutation_rates': self.mutation_rates.tolist(),
            'fitness_history': self.fitness_history,
            'generation_stats': self.generation_stats,
            'stagnation_counter': self.stagnation_counter
        }
        
        # Save checkpoint data
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save networks
        for i, network in enumerate(self.population):
            network.save(f"{filename}_network_{i}.json")
        
        # Save Hall of Fame
        self.hall_of_fame.save(f"{filename}_hall_of_fame.json")
        
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.fitness_scores = np.array(checkpoint['fitness_scores'])
        self.mutation_rates = np.array(checkpoint['mutation_rates'])
        self.fitness_history = checkpoint['fitness_history']
        self.generation_stats = checkpoint['generation_stats']
        self.stagnation_counter = checkpoint['stagnation_counter']
        
        # Load networks
        self.population = []
        for i in range(self.population_size):
            network = EnhancedYanivNN.load(f"{filename}_network_{i}.json")
            self.population.append(network)
        
        # Load Hall of Fame
        self.hall_of_fame.load(f"{filename}_hall_of_fame.json")
        
        print(f"Checkpoint loaded from {filename}")


def train_enhanced_ai(generations: int = 100, checkpoint_interval: int = 10):
    """Train the enhanced AI"""
    ga = AdaptiveGeneticAlgorithm(
        population_size=50,
        elite_size=5,
        mutation_rate_initial=0.1,
        crossover_rate=0.7
    )
    
    start_time = time.time()
    
    try:
        for gen in range(generations):
            ga.evolve_generation()
            
            # Save checkpoint
            if (gen + 1) % checkpoint_interval == 0:
                ga.save_checkpoint(f"enhanced_checkpoint_gen_{gen + 1}.pkl")
            
            # Early stopping if converged
            if ga.stagnation_counter > 20:
                print("Converged - stopping early")
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Create and save final ensemble
    ensemble = ga.create_ensemble(top_n=5)
    ensemble.save("enhanced_yaniv_ensemble.json")
    
    # Save final results
    best_idx = np.argmax(ga.fitness_scores)
    ga.population[best_idx].save("enhanced_yaniv_best.json")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds")
    print(f"Best fitness: {ga.fitness_scores[best_idx]:.3f}")
    
    # Save training history
    with open("enhanced_training_history.json", 'w') as f:
        json.dump({
            'fitness_history': ga.fitness_history,
            'generation_stats': ga.generation_stats,
            'training_time': elapsed_time
        }, f)
    
    return ga


if __name__ == "__main__":
    train_enhanced_ai(generations=50)