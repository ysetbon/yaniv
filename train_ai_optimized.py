#!/usr/bin/env python3
"""
Optimized Yaniv AI Training with parallel processing and vectorized operations
"""

from genetic_algorithm_optimized import GeneticAlgorithmOptimized
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import multiprocessing as mp
import psutil

def print_system_info():
    """Print system information for performance tracking"""
    print("\n=== System Information ===")
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory: {psutil.virtual_memory().percent}% used")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print("========================\n")

def benchmark_comparison():
    """Run a quick benchmark comparing original vs optimized"""
    print("\n=== Performance Benchmark ===")
    
    # Test parameters
    pop_size = 10
    games = 5
    
    # Import original modules for comparison
    try:
        from genetic_algorithm import GeneticAlgorithm
        from yaniv_neural_network import YanivNeuralNetwork
        
        # Benchmark original version
        print("Testing original implementation...")
        ga_original = GeneticAlgorithm(population_size=pop_size, top_k=3)
        ga_original.initialize_population()
        
        start_time = time.time()
        ga_original.evaluate_population(games_per_matchup=games)
        original_time = time.time() - start_time
        print(f"Original: {original_time:.2f}s")
    except:
        print("Original implementation not available for comparison")
        original_time = None
    
    # Benchmark optimized version
    print("Testing optimized implementation...")
    ga_optimized = GeneticAlgorithmOptimized(population_size=pop_size, top_k=3)
    ga_optimized.initialize_population()
    
    start_time = time.time()
    ga_optimized.evaluate_population_parallel(games_per_matchup=games)
    optimized_time = time.time() - start_time
    print(f"Optimized: {optimized_time:.2f}s")
    
    if original_time:
        speedup = original_time / optimized_time
        print(f"\nSpeedup: {speedup:.2f}x faster")
    
    print("===========================\n")

def main():
    parser = argparse.ArgumentParser(description='Train Yaniv AI using optimized genetic algorithm')
    parser.add_argument('--population', type=int, default=50, 
                      help='Population size (default: 50)')
    parser.add_argument('--generations', type=int, default=100,
                      help='Number of generations (default: 100)')
    parser.add_argument('--games', type=int, default=10,
                      help='Games per matchup (default: 10)')
    parser.add_argument('--top-k', type=int, default=10,
                      help='Number of top performers to keep (default: 10)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                      help='Mutation rate (default: 0.1)')
    parser.add_argument('--mutation-strength', type=float, default=0.1,
                      help='Mutation strength (default: 0.1)')
    parser.add_argument('--workers', type=int, default=None,
                      help='Number of parallel workers (default: all CPU cores)')
    parser.add_argument('--no-parallel', action='store_true',
                      help='Disable parallel processing')
    parser.add_argument('--benchmark', action='store_true',
                      help='Run performance benchmark')
    parser.add_argument('--resume', type=str, default=None,
                      help='Resume from checkpoint file')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='Save checkpoint every N generations (default: 10)')
    
    args = parser.parse_args()
    
    print("=== Optimized Yaniv AI Training ===")
    print(f"Configuration:")
    print(f"  Population size: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Games per matchup: {args.games}")
    print(f"  Top K survivors: {args.top_k}")
    print(f"  Mutation rate: {args.mutation_rate}")
    print(f"  Mutation strength: {args.mutation_strength}")
    print(f"  Parallel workers: {args.workers or 'auto'}")
    print(f"  Use parallel: {not args.no_parallel}")
    
    # Print system info
    print_system_info()
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_comparison()
    
    # Initialize genetic algorithm
    ga = GeneticAlgorithmOptimized(
        population_size=args.population,
        top_k=args.top_k,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        num_workers=args.workers
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ga.load_checkpoint(args.resume)
    else:
        # Initialize new population
        ga.initialize_population()
    
    # Calculate total games and time estimate
    total_games_per_gen = args.population * (args.population - 1) * args.games
    print(f"\nTotal games per generation: {total_games_per_gen:,}")
    
    # Estimate time based on CPU cores
    cpu_cores = args.workers or mp.cpu_count()
    estimated_games_per_sec = cpu_cores * 50  # Conservative estimate
    estimated_time_per_gen = total_games_per_gen / estimated_games_per_sec
    total_estimated_time = estimated_time_per_gen * args.generations
    
    print(f"Estimated time per generation: {estimated_time_per_gen:.1f} seconds")
    print(f"Total estimated time: {total_estimated_time/60:.1f} minutes")
    print("\n" + "="*50 + "\n")
    
    # Run evolution
    ga.evolve_parallel(
        num_generations=args.generations, 
        games_per_matchup=args.games,
        use_parallel=not args.no_parallel,
        save_interval=args.save_interval
    )
    
    # Plot fitness history with enhanced visualization
    if ga.best_fitness_history:
        plt.figure(figsize=(12, 8))
        
        # Main fitness plot
        plt.subplot(2, 1, 1)
        generations = range(len(ga.best_fitness_history))
        plt.plot(generations, ga.best_fitness_history, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (Win Rate)')
        plt.title('Yaniv AI Evolution Progress - Optimized Training')
        plt.grid(True, alpha=0.3)
        
        # Add moving average
        if len(ga.best_fitness_history) > 10:
            window = min(10, len(ga.best_fitness_history) // 4)
            moving_avg = np.convolve(ga.best_fitness_history, 
                                    np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(ga.best_fitness_history)), 
                    moving_avg, 'r--', alpha=0.7, label=f'{window}-gen moving average')
            plt.legend()
        
        # Improvement rate plot
        plt.subplot(2, 1, 2)
        if len(ga.best_fitness_history) > 1:
            improvements = np.diff(ga.best_fitness_history)
            plt.bar(range(1, len(ga.best_fitness_history)), improvements, 
                   color=['green' if x > 0 else 'red' for x in improvements], alpha=0.7)
            plt.xlabel('Generation')
            plt.ylabel('Fitness Improvement')
            plt.title('Generation-to-Generation Improvement')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('fitness_history_optimized.png', dpi=150)
        print("\nFitness history saved to fitness_history_optimized.png")
        
        # Print statistics
        print("\n=== Training Statistics ===")
        print(f"Initial fitness: {ga.best_fitness_history[0]:.3f}")
        print(f"Final fitness: {ga.best_fitness_history[-1]:.3f}")
        print(f"Peak fitness: {max(ga.best_fitness_history):.3f}")
        print(f"Average improvement per generation: {np.mean(np.diff(ga.best_fitness_history)):.4f}")
        
        # Find biggest improvements
        if len(ga.best_fitness_history) > 1:
            improvements = list(enumerate(np.diff(ga.best_fitness_history), 1))
            improvements.sort(key=lambda x: x[1], reverse=True)
            print(f"Biggest improvement: Generation {improvements[0][0]} (+{improvements[0][1]:.3f})")
    
    print("\n=== Training Complete! ===")
    print("Best network saved to: saved_networks/best_overall_optimized.json")
    print(f"Final checkpoint: saved_networks/checkpoint_gen_{ga.generation}.pkl")

if __name__ == "__main__":
    main()