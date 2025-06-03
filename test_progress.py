#!/usr/bin/env python3
"""Quick test of progress bar functionality"""

from genetic_algorithm_optimized import GeneticAlgorithmOptimized
import time

def test_progress():
    print("Testing progress bar with small population...")
    
    # Create small GA instance
    ga = GeneticAlgorithmOptimized(
        population_size=10,  # Small for quick test
        top_k=3,
        num_workers=4
    )
    
    # Initialize population
    ga.initialize_population()
    
    # Run one generation to see progress bar
    print("\nRunning evaluation with progress bar:")
    ga.evaluate_population_parallel(games_per_matchup=5)
    
    print("\nProgress bar test complete!")

if __name__ == "__main__":
    test_progress()