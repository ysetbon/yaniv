#!/usr/bin/env python3
"""
Train Yaniv AI using genetic algorithm
"""

from genetic_algorithm import GeneticAlgorithm
import argparse
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description='Train Yaniv AI using genetic algorithm')
    parser.add_argument('--population', type=int, default=30, 
                      help='Population size (default: 30)')
    parser.add_argument('--generations', type=int, default=50,
                      help='Number of generations (default: 50)')
    parser.add_argument('--games', type=int, default=10,
                      help='Games per matchup (default: 10)')
    parser.add_argument('--top-k', type=int, default=10,
                      help='Number of top performers to keep (default: 10)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                      help='Mutation rate (default: 0.1)')
    parser.add_argument('--mutation-strength', type=float, default=0.1,
                      help='Mutation strength (default: 0.1)')
    
    args = parser.parse_args()
    
    print("Starting Yaniv AI Training")
    print(f"Configuration:")
    print(f"  Population size: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Games per matchup: {args.games}")
    print(f"  Top K survivors: {args.top_k}")
    print(f"  Mutation rate: {args.mutation_rate}")
    print(f"  Mutation strength: {args.mutation_strength}")
    print()
    
    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        population_size=args.population,
        top_k=args.top_k,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength
    )
    
    # Initialize population
    ga.initialize_population()
    
    # Run evolution
    ga.evolve(args.generations, args.games)
    
    # Plot fitness history
    if ga.best_fitness_history:
        plt.figure(figsize=(10, 6))
        plt.plot(ga.best_fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (Win Rate)')
        plt.title('Yaniv AI Evolution Progress')
        plt.grid(True)
        plt.savefig('fitness_history.png')
        print("\nFitness history saved to fitness_history.png")
    
    print("\nTraining complete!")
    print("Best network saved to: saved_networks/best_overall.json")

if __name__ == "__main__":
    main()