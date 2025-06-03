import numpy as np
from typing import List, Tuple
from yaniv_neural_network import YanivNeuralNetwork
from yaniv_game_ai import YanivGameAI
import random
import json
import os

class GeneticAlgorithm:
    """Genetic algorithm for evolving Yaniv AI players"""
    
    def __init__(self, population_size: int = 50, top_k: int = 10, 
                 mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        self.population_size = population_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.generation = 0
        self.population: List[YanivNeuralNetwork] = []
        self.best_fitness_history = []
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        for i in range(self.population_size):
            network = YanivNeuralNetwork(network_id=i)
            self.population.append(network)
        print(f"Initialized population with {self.population_size} neural networks")
    
    def evaluate_population(self, games_per_matchup: int = 10):
        """Run tournament between all AI players"""
        # Reset stats for all networks
        for network in self.population:
            network.fitness = 0
            network.wins = 0
            network.games_played = 0
        
        total_matchups = self.population_size * (self.population_size - 1) // 2
        matchup_count = 0
        
        # Round-robin tournament
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                player1 = self.population[i]
                player2 = self.population[j]
                
                # Play multiple games between these two players
                p1_wins = 0
                p2_wins = 0
                
                for game_num in range(games_per_matchup):
                    # Alternate who goes first
                    if game_num % 2 == 0:
                        winner = self.simulate_game(player1, player2)
                    else:
                        winner = self.simulate_game(player2, player1)
                    
                    if winner == player1:
                        p1_wins += 1
                    else:
                        p2_wins += 1
                
                # Update stats
                player1.games_played += games_per_matchup
                player2.games_played += games_per_matchup
                player1.wins += p1_wins
                player2.wins += p2_wins
                
                matchup_count += 1
                if matchup_count % 10 == 0:
                    print(f"Progress: {matchup_count}/{total_matchups} matchups complete")
        
        # Calculate fitness (win rate)
        for network in self.population:
            if network.games_played > 0:
                network.fitness = network.wins / network.games_played
    
    def simulate_game(self, player1: YanivNeuralNetwork, 
                     player2: YanivNeuralNetwork) -> YanivNeuralNetwork:
        """Simulate a single game between two AI players"""
        game = YanivGameAI()
        winner = game.play_game([player1, player2])
        return winner
    
    def select_survivors(self) -> List[YanivNeuralNetwork]:
        """Select top performing networks"""
        # Sort by fitness (win rate)
        sorted_population = sorted(self.population, 
                                 key=lambda x: x.fitness, 
                                 reverse=True)
        
        # Keep top K
        survivors = sorted_population[:self.top_k]
        
        print(f"\nGeneration {self.generation} results:")
        print(f"Best fitness: {survivors[0].fitness:.3f}")
        print(f"Top {self.top_k} fitness scores: ", 
              [f"{s.fitness:.3f}" for s in survivors])
        
        return survivors
    
    def create_next_generation(self, survivors: List[YanivNeuralNetwork]):
        """Create new generation from survivors"""
        new_population = []
        
        # Keep the survivors
        for i, survivor in enumerate(survivors):
            survivor.network_id = i
            new_population.append(survivor)
        
        # Create mutated versions
        mutations_per_survivor = 3
        current_id = len(survivors)
        
        for survivor in survivors:
            for _ in range(mutations_per_survivor):
                if current_id < self.population_size:
                    mutant = survivor.copy()
                    mutant.network_id = current_id
                    mutant.mutate(self.mutation_rate, self.mutation_strength)
                    new_population.append(mutant)
                    current_id += 1
        
        # Fill remaining slots with new random networks
        while len(new_population) < self.population_size:
            network = YanivNeuralNetwork(network_id=current_id)
            new_population.append(network)
            current_id += 1
        
        self.population = new_population
        self.generation += 1
    
    def evolve(self, num_generations: int, games_per_matchup: int = 10):
        """Run the evolutionary process"""
        print(f"Starting evolution for {num_generations} generations")
        print(f"Population size: {self.population_size}")
        print(f"Games per matchup: {games_per_matchup}")
        print(f"Total games per generation: {self.population_size * (self.population_size - 1) * games_per_matchup}")
        
        for gen in range(num_generations):
            print(f"\n--- Generation {self.generation} ---")
            
            # Evaluate current population
            self.evaluate_population(games_per_matchup)
            
            # Select survivors
            survivors = self.select_survivors()
            
            # Track best fitness
            self.best_fitness_history.append(survivors[0].fitness)
            
            # Save best network
            self.save_best_network(survivors[0])
            
            # Create next generation
            if gen < num_generations - 1:
                self.create_next_generation(survivors)
        
        print("\nEvolution complete!")
        print(f"Best fitness over time: {self.best_fitness_history}")
    
    def save_best_network(self, network: YanivNeuralNetwork):
        """Save the best performing network"""
        if not os.path.exists('saved_networks'):
            os.makedirs('saved_networks')
        
        filename = f'saved_networks/best_gen_{self.generation}.json'
        network.save(filename)
        
        # Also save as 'best_overall' if it's the best we've seen
        if not hasattr(self, 'best_overall_fitness') or network.fitness > self.best_overall_fitness:
            self.best_overall_fitness = network.fitness
            network.save('saved_networks/best_overall.json')
            print(f"New best network saved with fitness: {network.fitness:.3f}")
    
    def load_population(self, generation: int):
        """Load a previously saved population"""
        # Implementation for loading saved populations
        pass