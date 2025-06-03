import numpy as np
from typing import List, Tuple, Dict, Optional
from yaniv_neural_network_optimized import YanivNeuralNetworkOptimized
from yaniv_game_ai_optimized import YanivGameAIOptimized
from parallel_executor_simple import SimpleParallelExecutor
import random
import json
import os
import time
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import gc

class GeneticAlgorithmOptimized:
    """Optimized genetic algorithm with parallel evaluation and vectorized operations"""
    
    def __init__(self, population_size: int = 50, top_k: int = 10, 
                 mutation_rate: float = 0.1, mutation_strength: float = 0.1,
                 num_workers: Optional[int] = None):
        self.population_size = population_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.generation = 0
        self.population: List[YanivNeuralNetworkOptimized] = []
        self.best_fitness_history = []
        self.num_workers = num_workers or mp.cpu_count()
        self.parallel_executor = SimpleParallelExecutor(self.num_workers)
        
        # Memory optimization: reuse population arrays
        self.fitness_array = np.zeros(population_size)
        self.wins_array = np.zeros(population_size)
        self.games_array = np.zeros(population_size)
        
    def initialize_population(self):
        """Create initial random population with optimized networks"""
        self.population = []
        for i in range(self.population_size):
            network = YanivNeuralNetworkOptimized(network_id=i)
            self.population.append(network)
        print(f"Initialized population with {self.population_size} optimized neural networks")
        print(f"Using {self.num_workers} CPU cores for parallel evaluation")
    
    def evaluate_population_parallel(self, games_per_matchup: int = 10):
        """Run tournament in parallel using all CPU cores"""
        print(f"Evaluating population using {self.num_workers} parallel workers...")
        start_time = time.time()
        
        # Reset stats arrays
        self.fitness_array.fill(0)
        self.wins_array.fill(0)
        self.games_array.fill(0)
        
        # Run parallel tournament with progress bar
        results = self.parallel_executor.play_tournament_parallel_with_progress(
            self.population, games_per_matchup
        )
        
        # Update population stats from results
        for network_id, (wins, games) in results.items():
            self.wins_array[network_id] = wins
            self.games_array[network_id] = games
            if games > 0:
                self.fitness_array[network_id] = wins / games
        
        # Update network objects
        for i, network in enumerate(self.population):
            network.wins = int(self.wins_array[i])
            network.games_played = int(self.games_array[i])
            network.fitness = self.fitness_array[i]
        
        elapsed = time.time() - start_time
        total_games = self.population_size * (self.population_size - 1) * games_per_matchup
        print(f"Evaluation complete in {elapsed:.2f}s ({total_games / elapsed:.1f} games/sec)")
    
    def evaluate_population_vectorized(self, games_per_matchup: int = 10):
        """Alternative: Batch evaluation with vectorized neural network operations"""
        print("Evaluating population with vectorized operations...")
        
        # Reset stats
        for network in self.population:
            network.fitness = 0
            network.wins = 0
            network.games_played = 0
        
        total_matchups = self.population_size * (self.population_size - 1) // 2
        
        # Create batches of games for vectorized processing
        batch_size = min(32, self.population_size)  # Process multiple games simultaneously
        
        with tqdm(total=total_matchups, desc="Matchups") as pbar:
            # Process matchups in batches
            for batch_start in range(0, len(self.population), batch_size):
                batch_end = min(batch_start + batch_size, len(self.population))
                batch_players = self.population[batch_start:batch_end]
                
                # Play games for this batch against all other players
                for opponent_idx in range(len(self.population)):
                    if opponent_idx < batch_start or opponent_idx >= batch_end:
                        opponent = self.population[opponent_idx]
                        
                        # Simulate games in batch
                        for player_idx, player in enumerate(batch_players):
                            if batch_start + player_idx != opponent_idx:
                                wins = self._simulate_games_vectorized(player, opponent, games_per_matchup)
                                
                                player.wins += wins
                                player.games_played += games_per_matchup
                                opponent.wins += (games_per_matchup - wins)
                                opponent.games_played += games_per_matchup
                                
                                pbar.update(0.5)  # Half matchup done
        
        # Calculate fitness
        for network in self.population:
            if network.games_played > 0:
                network.fitness = network.wins / network.games_played
    
    def _simulate_games_vectorized(self, player1: YanivNeuralNetworkOptimized, 
                                 player2: YanivNeuralNetworkOptimized, 
                                 num_games: int) -> int:
        """Simulate multiple games using vectorized operations where possible"""
        p1_wins = 0
        
        # Play games in small batches for memory efficiency
        games_per_batch = min(4, num_games)
        
        for batch_start in range(0, num_games, games_per_batch):
            batch_end = min(batch_start + games_per_batch, num_games)
            batch_size = batch_end - batch_start
            
            # Simulate games in this batch
            for game_idx in range(batch_size):
                # Alternate starting player
                if (batch_start + game_idx) % 2 == 0:
                    players = [player1, player2]
                else:
                    players = [player2, player1]
                
                game = YanivGameAIOptimized()
                winner = game.play_game(players)
                
                if winner.network_id == player1.network_id:
                    p1_wins += 1
        
        return p1_wins
    
    def select_survivors_vectorized(self) -> List[YanivNeuralNetworkOptimized]:
        """Select top performers using vectorized operations"""
        # Use numpy for fast sorting
        sorted_indices = np.argsort(self.fitness_array)[::-1]  # Descending order
        
        # Select top K
        top_indices = sorted_indices[:self.top_k]
        survivors = [self.population[idx] for idx in top_indices]
        
        # Print results
        print(f"\nGeneration {self.generation} results:")
        print(f"Best fitness: {self.fitness_array[top_indices[0]]:.3f}")
        print(f"Top {self.top_k} fitness scores: ", 
              [f"{self.fitness_array[idx]:.3f}" for idx in top_indices[:self.top_k]])
        
        # Statistics
        mean_fitness = np.mean(self.fitness_array)
        std_fitness = np.std(self.fitness_array)
        print(f"Population fitness: mean={mean_fitness:.3f}, std={std_fitness:.3f}")
        
        return survivors
    
    def create_next_generation_optimized(self, survivors: List[YanivNeuralNetworkOptimized]):
        """Create new generation with memory optimization"""
        new_population = []
        
        # Keep the survivors (elitism)
        for i, survivor in enumerate(survivors):
            survivor.network_id = i
            new_population.append(survivor)
        
        # Tournament selection for breeding
        tournament_size = 3
        num_offspring = self.population_size - len(survivors)
        
        current_id = len(survivors)
        
        # Create offspring through tournament selection and mutation
        for _ in range(num_offspring):
            # Tournament selection
            tournament_indices = np.random.choice(len(survivors), tournament_size, replace=False)
            tournament_fitness = [survivors[idx].fitness for idx in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parent = survivors[winner_idx]
            
            # Create mutated offspring
            offspring = parent.copy()
            offspring.network_id = current_id
            offspring.mutate_vectorized(self.mutation_rate, self.mutation_strength)
            
            # Optionally add some fully random networks for diversity
            if random.random() < 0.1:  # 10% chance of random network
                offspring = YanivNeuralNetworkOptimized(network_id=current_id)
            
            new_population.append(offspring)
            current_id += 1
        
        self.population = new_population
        self.generation += 1
        
        # Force garbage collection to free memory
        gc.collect()
    
    def adaptive_mutation_rate(self):
        """Adapt mutation rate based on population diversity"""
        if len(self.best_fitness_history) > 5:
            # Check if fitness is plateauing
            recent_fitness = self.best_fitness_history[-5:]
            fitness_variance = np.var(recent_fitness)
            
            if fitness_variance < 0.001:  # Plateau detected
                # Increase mutation rate to explore more
                self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
                self.mutation_strength = min(0.3, self.mutation_strength * 1.2)
                print(f"Plateau detected - increasing mutation rate to {self.mutation_rate:.3f}")
            else:
                # Decrease mutation rate for exploitation
                self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
                self.mutation_strength = max(0.05, self.mutation_strength * 0.95)
    
    def evolve_parallel(self, num_generations: int, games_per_matchup: int = 10,
                       use_parallel: bool = True, save_interval: int = 5):
        """Run the evolutionary process with parallel evaluation"""
        print(f"Starting parallel evolution for {num_generations} generations")
        print(f"Population size: {self.population_size}")
        print(f"Games per matchup: {games_per_matchup}")
        print(f"CPU cores: {self.num_workers}")
        print(f"Total games per generation: {self.population_size * (self.population_size - 1) * games_per_matchup}")
        
        start_time = time.time()
        
        for gen in range(num_generations):
            print(f"\n--- Generation {self.generation} ---")
            gen_start = time.time()
            
            # Evaluate current population
            if use_parallel:
                self.evaluate_population_parallel(games_per_matchup)
            else:
                self.evaluate_population_vectorized(games_per_matchup)
            
            # Select survivors
            survivors = self.select_survivors_vectorized()
            
            # Track best fitness
            self.best_fitness_history.append(survivors[0].fitness)
            
            # Save best network
            self.save_best_network(survivors[0])
            
            # Save checkpoint periodically
            if (gen + 1) % save_interval == 0:
                self.save_checkpoint()
            
            # Adaptive mutation rate
            self.adaptive_mutation_rate()
            
            # Create next generation
            if gen < num_generations - 1:
                self.create_next_generation_optimized(survivors)
            
            gen_time = time.time() - gen_start
            print(f"Generation completed in {gen_time:.2f}s")
            
            # Estimate time remaining
            if gen > 0:
                avg_gen_time = (time.time() - start_time) / (gen + 1)
                remaining_time = avg_gen_time * (num_generations - gen - 1)
                print(f"Estimated time remaining: {remaining_time/60:.1f} minutes")
        
        total_time = time.time() - start_time
        print(f"\nEvolution complete in {total_time/60:.1f} minutes!")
        print(f"Best fitness achieved: {max(self.best_fitness_history):.3f}")
        print(f"Best fitness over time: {self.best_fitness_history}")
    
    def save_best_network(self, network: YanivNeuralNetworkOptimized):
        """Save the best performing network"""
        if not os.path.exists('saved_networks'):
            os.makedirs('saved_networks')
        
        filename = f'saved_networks/best_gen_{self.generation}.json'
        network.save(filename)
        
        # Also save as 'best_overall' if it's the best we've seen
        if not hasattr(self, 'best_overall_fitness') or network.fitness > self.best_overall_fitness:
            self.best_overall_fitness = network.fitness
            network.save('saved_networks/best_overall_optimized.json')
            print(f"New best network saved with fitness: {network.fitness:.3f}")
    
    def save_checkpoint(self):
        """Save entire population state for resuming training"""
        checkpoint = {
            'generation': self.generation,
            'population_size': self.population_size,
            'top_k': self.top_k,
            'mutation_rate': self.mutation_rate,
            'mutation_strength': self.mutation_strength,
            'best_fitness_history': self.best_fitness_history,
            'population_weights': []
        }
        
        # Save all network weights
        for network in self.population:
            checkpoint['population_weights'].append({
                'network_id': network.network_id,
                'w1': network.w1.tolist(),
                'b1': network.b1.tolist(),
                'w2': network.w2.tolist(),
                'b2': network.b2.tolist(),
                'fitness': network.fitness,
                'wins': network.wins,
                'games_played': network.games_played
            })
        
        filename = f'saved_networks/checkpoint_gen_{self.generation}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load population state from checkpoint"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint['generation']
        self.population_size = checkpoint['population_size']
        self.top_k = checkpoint['top_k']
        self.mutation_rate = checkpoint['mutation_rate']
        self.mutation_strength = checkpoint['mutation_strength']
        self.best_fitness_history = checkpoint['best_fitness_history']
        
        # Restore population
        self.population = []
        for weights in checkpoint['population_weights']:
            network = YanivNeuralNetworkOptimized(weights['network_id'])
            network.w1 = np.array(weights['w1'])
            network.b1 = np.array(weights['b1'])
            network.w2 = np.array(weights['w2'])
            network.b2 = np.array(weights['b2'])
            network.fitness = weights['fitness']
            network.wins = weights['wins']
            network.games_played = weights['games_played']
            self.population.append(network)
        
        print(f"Checkpoint loaded from generation {self.generation}")