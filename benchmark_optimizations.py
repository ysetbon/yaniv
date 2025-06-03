#!/usr/bin/env python3
"""
Benchmark script to compare original vs optimized Yaniv AI training performance
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import multiprocessing as mp
import psutil
import os
import gc

# Import both versions
from yaniv_neural_network import YanivNeuralNetwork
from yaniv_neural_network_optimized import YanivNeuralNetworkOptimized
from yaniv_game_ai import YanivGameAI
from yaniv_game_ai_optimized import YanivGameAIOptimized, ParallelGameExecutor
from genetic_algorithm import GeneticAlgorithm
from genetic_algorithm_optimized import GeneticAlgorithmOptimized


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def __init__(self):
        self.results = {
            'neural_network': {},
            'game_simulation': {},
            'population_evaluation': {},
            'memory_usage': {}
        }
        
    def benchmark_neural_network_operations(self, num_iterations: int = 1000):
        """Benchmark neural network forward pass and mutations"""
        print("\n=== Neural Network Operations Benchmark ===")
        
        # Create test data
        test_game_state = {
            'hand': [{'value': i, 'suit': 'hearts'} for i in range(1, 6)],
            'hand_value': 15,
            'deck_size': 40,
            'opponent_cards': [5, 5, 5],
            'last_discard': {'value': 7, 'suit': 'diamonds'}
        }
        
        # Test original implementation
        print("Testing original neural network...")
        nn_original = YanivNeuralNetwork(0)
        
        # Forward pass benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            nn_original.forward(test_game_state)
        original_forward_time = time.time() - start_time
        
        # Mutation benchmark
        start_time = time.time()
        for _ in range(100):
            nn_original.mutate(0.1, 0.1)
        original_mutate_time = time.time() - start_time
        
        # Test optimized implementation
        print("Testing optimized neural network...")
        nn_optimized = YanivNeuralNetworkOptimized(0)
        
        # Forward pass benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            nn_optimized.forward(test_game_state)
        optimized_forward_time = time.time() - start_time
        
        # Batch forward pass benchmark
        batch_states = [test_game_state] * 32
        start_time = time.time()
        for _ in range(num_iterations // 32):
            nn_optimized.forward_batch(batch_states)
        batch_forward_time = time.time() - start_time
        
        # Mutation benchmark
        start_time = time.time()
        for _ in range(100):
            nn_optimized.mutate_vectorized(0.1, 0.1)
        optimized_mutate_time = time.time() - start_time
        
        # Store results
        self.results['neural_network'] = {
            'original_forward': original_forward_time,
            'optimized_forward': optimized_forward_time,
            'batch_forward': batch_forward_time,
            'original_mutate': original_mutate_time,
            'optimized_mutate': optimized_mutate_time,
            'forward_speedup': original_forward_time / optimized_forward_time,
            'batch_speedup': (original_forward_time * 32) / batch_forward_time,
            'mutate_speedup': original_mutate_time / optimized_mutate_time
        }
        
        print(f"Forward pass speedup: {self.results['neural_network']['forward_speedup']:.2f}x")
        print(f"Batch forward speedup: {self.results['neural_network']['batch_speedup']:.2f}x")
        print(f"Mutation speedup: {self.results['neural_network']['mutate_speedup']:.2f}x")
        
    def benchmark_game_simulation(self, num_games: int = 100):
        """Benchmark game simulation performance"""
        print("\n=== Game Simulation Benchmark ===")
        
        # Create test players
        players_original = [YanivNeuralNetwork(i) for i in range(2)]
        players_optimized = [YanivNeuralNetworkOptimized(i) for i in range(2)]
        
        # Test original implementation
        print("Testing original game simulation...")
        game_original = YanivGameAI()
        
        start_time = time.time()
        for _ in range(num_games):
            game_original.play_game(players_original)
        original_game_time = time.time() - start_time
        
        # Test optimized implementation
        print("Testing optimized game simulation...")
        game_optimized = YanivGameAIOptimized()
        
        start_time = time.time()
        for _ in range(num_games):
            game_optimized.play_game(players_optimized)
        optimized_game_time = time.time() - start_time
        
        # Store results
        self.results['game_simulation'] = {
            'original_time': original_game_time,
            'optimized_time': optimized_game_time,
            'speedup': original_game_time / optimized_game_time,
            'games_per_sec_original': num_games / original_game_time,
            'games_per_sec_optimized': num_games / optimized_game_time
        }
        
        print(f"Game simulation speedup: {self.results['game_simulation']['speedup']:.2f}x")
        print(f"Original: {self.results['game_simulation']['games_per_sec_original']:.1f} games/sec")
        print(f"Optimized: {self.results['game_simulation']['games_per_sec_optimized']:.1f} games/sec")
        
    def benchmark_population_evaluation(self, pop_size: int = 20, games_per_matchup: int = 5):
        """Benchmark population evaluation with and without parallelization"""
        print("\n=== Population Evaluation Benchmark ===")
        
        # Test original implementation
        print("Testing original population evaluation...")
        ga_original = GeneticAlgorithm(population_size=pop_size, top_k=5)
        ga_original.initialize_population()
        
        start_time = time.time()
        ga_original.evaluate_population(games_per_matchup)
        original_eval_time = time.time() - start_time
        
        # Test optimized sequential
        print("Testing optimized sequential evaluation...")
        ga_opt_seq = GeneticAlgorithmOptimized(population_size=pop_size, top_k=5, num_workers=1)
        ga_opt_seq.initialize_population()
        
        start_time = time.time()
        ga_opt_seq.evaluate_population_vectorized(games_per_matchup)
        opt_seq_eval_time = time.time() - start_time
        
        # Test optimized parallel
        print(f"Testing optimized parallel evaluation ({mp.cpu_count()} cores)...")
        ga_opt_par = GeneticAlgorithmOptimized(population_size=pop_size, top_k=5)
        ga_opt_par.initialize_population()
        
        start_time = time.time()
        ga_opt_par.evaluate_population_parallel(games_per_matchup)
        opt_par_eval_time = time.time() - start_time
        
        # Calculate total games
        total_games = pop_size * (pop_size - 1) * games_per_matchup
        
        # Store results
        self.results['population_evaluation'] = {
            'original_time': original_eval_time,
            'optimized_seq_time': opt_seq_eval_time,
            'optimized_par_time': opt_par_eval_time,
            'seq_speedup': original_eval_time / opt_seq_eval_time,
            'par_speedup': original_eval_time / opt_par_eval_time,
            'parallel_efficiency': opt_seq_eval_time / (opt_par_eval_time * mp.cpu_count()),
            'games_per_sec_original': total_games / original_eval_time,
            'games_per_sec_opt_seq': total_games / opt_seq_eval_time,
            'games_per_sec_opt_par': total_games / opt_par_eval_time
        }
        
        print(f"Sequential optimization speedup: {self.results['population_evaluation']['seq_speedup']:.2f}x")
        print(f"Parallel optimization speedup: {self.results['population_evaluation']['par_speedup']:.2f}x")
        print(f"Parallel efficiency: {self.results['population_evaluation']['parallel_efficiency']:.2%}")
        print(f"Games/sec - Original: {self.results['population_evaluation']['games_per_sec_original']:.1f}")
        print(f"Games/sec - Optimized Sequential: {self.results['population_evaluation']['games_per_sec_opt_seq']:.1f}")
        print(f"Games/sec - Optimized Parallel: {self.results['population_evaluation']['games_per_sec_opt_par']:.1f}")
        
    def benchmark_memory_usage(self, pop_size: int = 50):
        """Benchmark memory usage of different implementations"""
        print("\n=== Memory Usage Benchmark ===")
        
        # Force garbage collection
        gc.collect()
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test original implementation
        print("Testing original memory usage...")
        networks_original = [YanivNeuralNetwork(i) for i in range(pop_size)]
        original_memory = process.memory_info().rss / 1024 / 1024 - baseline_memory
        
        # Clear
        del networks_original
        gc.collect()
        
        # Test optimized implementation
        print("Testing optimized memory usage...")
        networks_optimized = [YanivNeuralNetworkOptimized(i) for i in range(pop_size)]
        optimized_memory = process.memory_info().rss / 1024 / 1024 - baseline_memory
        
        self.results['memory_usage'] = {
            'original_mb': original_memory,
            'optimized_mb': optimized_memory,
            'memory_reduction': 1 - (optimized_memory / original_memory) if original_memory > 0 else 0
        }
        
        print(f"Original memory usage: {original_memory:.2f} MB")
        print(f"Optimized memory usage: {optimized_memory:.2f} MB")
        print(f"Memory reduction: {self.results['memory_usage']['memory_reduction']:.1%}")
        
    def plot_results(self):
        """Create visualization of benchmark results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Neural Network Operations
        nn_labels = ['Forward Pass', 'Batch Forward', 'Mutation']
        nn_speedups = [
            self.results['neural_network']['forward_speedup'],
            self.results['neural_network']['batch_speedup'],
            self.results['neural_network']['mutate_speedup']
        ]
        ax1.bar(nn_labels, nn_speedups, color=['blue', 'green', 'orange'])
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('Neural Network Operations Speedup')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        for i, v in enumerate(nn_speedups):
            ax1.text(i, v + 0.1, f'{v:.2f}x', ha='center')
        
        # Game Simulation Performance
        game_data = self.results['game_simulation']
        categories = ['Original', 'Optimized']
        games_per_sec = [game_data['games_per_sec_original'], game_data['games_per_sec_optimized']]
        ax2.bar(categories, games_per_sec, color=['red', 'green'])
        ax2.set_ylabel('Games per Second')
        ax2.set_title('Game Simulation Performance')
        for i, v in enumerate(games_per_sec):
            ax2.text(i, v + 1, f'{v:.1f}', ha='center')
        
        # Population Evaluation Comparison
        pop_data = self.results['population_evaluation']
        eval_methods = ['Original', 'Optimized\nSequential', 'Optimized\nParallel']
        eval_times = [
            pop_data['original_time'],
            pop_data['optimized_seq_time'],
            pop_data['optimized_par_time']
        ]
        ax3.bar(eval_methods, eval_times, color=['red', 'yellow', 'green'])
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Population Evaluation Time')
        for i, v in enumerate(eval_times):
            ax3.text(i, v + 0.1, f'{v:.2f}s', ha='center')
        
        # Overall Speedup Summary
        speedup_categories = ['NN Forward', 'NN Batch', 'Game Sim', 'Pop Eval\n(Parallel)']
        speedups = [
            self.results['neural_network']['forward_speedup'],
            self.results['neural_network']['batch_speedup'],
            self.results['game_simulation']['speedup'],
            self.results['population_evaluation']['par_speedup']
        ]
        colors = ['blue', 'green', 'orange', 'purple']
        ax4.bar(speedup_categories, speedups, color=colors)
        ax4.set_ylabel('Speedup Factor')
        ax4.set_title('Overall Performance Improvements')
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
        for i, v in enumerate(speedups):
            ax4.text(i, v + 0.5, f'{v:.1f}x', ha='center')
        
        plt.tight_layout()
        plt.savefig('optimization_benchmark_results.png', dpi=150)
        print("\nBenchmark results saved to optimization_benchmark_results.png")
        
    def generate_report(self):
        """Generate a comprehensive performance report"""
        print("\n" + "="*60)
        print("YANIV AI OPTIMIZATION PERFORMANCE REPORT")
        print("="*60)
        
        print("\n1. NEURAL NETWORK OPTIMIZATIONS:")
        print(f"   - Forward pass: {self.results['neural_network']['forward_speedup']:.2f}x faster")
        print(f"   - Batch processing: {self.results['neural_network']['batch_speedup']:.2f}x faster")
        print(f"   - Mutations: {self.results['neural_network']['mutate_speedup']:.2f}x faster")
        
        print("\n2. GAME SIMULATION:")
        print(f"   - Single game speedup: {self.results['game_simulation']['speedup']:.2f}x")
        print(f"   - Throughput improvement: {self.results['game_simulation']['games_per_sec_original']:.1f} → {self.results['game_simulation']['games_per_sec_optimized']:.1f} games/sec")
        
        print("\n3. POPULATION EVALUATION:")
        print(f"   - Sequential optimization: {self.results['population_evaluation']['seq_speedup']:.2f}x faster")
        print(f"   - Parallel optimization: {self.results['population_evaluation']['par_speedup']:.2f}x faster")
        print(f"   - Parallel efficiency: {self.results['population_evaluation']['parallel_efficiency']:.1%}")
        print(f"   - CPU cores utilized: {mp.cpu_count()}")
        
        print("\n4. MEMORY EFFICIENCY:")
        print(f"   - Original: {self.results['memory_usage']['original_mb']:.2f} MB")
        print(f"   - Optimized: {self.results['memory_usage']['optimized_mb']:.2f} MB")
        print(f"   - Reduction: {self.results['memory_usage']['memory_reduction']:.1%}")
        
        # Calculate overall speedup
        overall_speedup = self.results['population_evaluation']['par_speedup']
        print(f"\n5. OVERALL TRAINING SPEEDUP: {overall_speedup:.1f}x")
        
        print("\n6. KEY OPTIMIZATION TECHNIQUES IMPLEMENTED:")
        print("   ✓ Vectorized neural network operations (NumPy)")
        print("   ✓ Batch processing for forward passes")
        print("   ✓ JIT compilation for critical functions (Numba)")
        print("   ✓ Parallel game execution (multiprocessing)")
        print("   ✓ Memory-efficient array representations")
        print("   ✓ Pre-allocated arrays for reduced allocation overhead")
        print("   ✓ Optimized mutation operations")
        print("   ✓ Efficient hand value calculations")
        
        print("\n" + "="*60)


def main():
    """Run comprehensive benchmarks"""
    print("Starting Yaniv AI Optimization Benchmarks...")
    print(f"System: {mp.cpu_count()} CPU cores, {psutil.virtual_memory().total / (1024**3):.1f} GB RAM")
    
    benchmark = PerformanceBenchmark()
    
    # Run all benchmarks
    benchmark.benchmark_neural_network_operations(num_iterations=1000)
    benchmark.benchmark_game_simulation(num_games=100)
    benchmark.benchmark_population_evaluation(pop_size=20, games_per_matchup=5)
    benchmark.benchmark_memory_usage(pop_size=50)
    
    # Generate visualizations and report
    benchmark.plot_results()
    benchmark.generate_report()
    
    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()