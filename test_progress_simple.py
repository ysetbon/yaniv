#!/usr/bin/env python3
"""Test progress bar with smaller parameters"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genetic_algorithm_optimized import GeneticAlgorithmOptimized

def main():
    print("Testing optimized training with progress bar...")
    print("Using small population for quick test\n")
    
    # Small parameters for testing
    ga = GeneticAlgorithmOptimized(
        population_size=10,  # Very small
        top_k=3,
        num_workers=4
    )
    
    # Initialize and run one generation
    ga.initialize_population()
    
    print("\nRunning one generation to test progress bar:")
    ga.evolve_parallel(
        num_generations=1,
        games_per_matchup=5,
        use_parallel=True
    )
    
    print("\nTest complete! Progress bar should have appeared above.")

if __name__ == "__main__":
    main()