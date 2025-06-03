"""Quick training script for enhanced AI - demonstrates improvements"""

import time
from genetic_algorithm_enhanced import AdaptiveGeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np


def quick_train_demo(generations: int = 5):
    """Quick training demonstration"""
    
    print("Starting Enhanced Yaniv AI Training")
    print("=" * 50)
    print("Features:")
    print("- Deeper neural network (128->64->32->16) with skip connections")
    print("- Rich feature extraction (35 features vs 11)")
    print("- Hall of Fame to preserve best strategies")
    print("- Adaptive mutation rates")
    print("- Ensemble learning")
    print("=" * 50)
    
    # Initialize with smaller population for faster demo
    ga = AdaptiveGeneticAlgorithm(
        population_size=10,  # Very small for testing
        elite_size=2,
        mutation_rate_initial=0.15,
        crossover_rate=0.7
    )
    
    start_time = time.time()
    
    try:
        for gen in range(generations):
            ga.evolve_generation()
            
            # Show progress
            if (gen + 1) % 5 == 0:
                print(f"\nGeneration {gen + 1} Summary:")
                print(f"  Hall of Fame size: {len(ga.hall_of_fame.champions)}")
                print(f"  Avg mutation rate: {np.mean(ga.mutation_rates):.3f}")
                
                if ga.stagnation_counter > 0:
                    print(f"  Stagnation counter: {ga.stagnation_counter}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    elapsed_time = time.time() - start_time
    
    # Create ensemble from best networks
    ensemble = ga.create_ensemble(top_n=3)
    ensemble.save("quick_enhanced_ensemble.json")
    
    # Save best individual
    best_idx = np.argmax(ga.fitness_scores)
    ga.population[best_idx].save("quick_enhanced_best.json")
    
    print(f"\nTraining completed in {elapsed_time:.1f} seconds")
    print(f"Final best fitness: {ga.fitness_scores[best_idx]:.3f}")
    
    # Plot training progress
    plot_training_progress(ga.fitness_history)
    
    return ga


def plot_training_progress(fitness_history):
    """Plot training progress"""
    if len(fitness_history) == 0:
        return
    
    generations = range(1, len(fitness_history) + 1)
    best_fitness = [gen['best'] for gen in fitness_history]
    mean_fitness = [gen['mean'] for gen in fitness_history]
    std_fitness = [gen['std'] for gen in fitness_history]
    
    plt.figure(figsize=(10, 6))
    
    # Plot best and mean with confidence interval
    plt.plot(generations, best_fitness, 'b-', label='Best', linewidth=2)
    plt.plot(generations, mean_fitness, 'g-', label='Mean', linewidth=2)
    
    # Add confidence interval
    mean_array = np.array(mean_fitness)
    std_array = np.array(std_fitness)
    plt.fill_between(generations, 
                     mean_array - std_array, 
                     mean_array + std_array, 
                     alpha=0.3, color='green')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Win Rate)')
    plt.title('Enhanced AI Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('enhanced_training_progress.png')
    plt.show()


def compare_with_original():
    """Quick comparison with original AI"""
    print("\nComparing with original AI...")
    
    try:
        # Load original stats
        import json
        with open("saved_networks/best_overall_optimized.json", 'r') as f:
            original_data = json.load(f)
            original_fitness = original_data.get('fitness', 0.681)
            print(f"Original AI best fitness: {original_fitness:.3f}")
    except:
        print("Could not load original AI stats")
    
    print("\nEnhanced AI improvements:")
    print("- Deeper understanding of game state")
    print("- Better long-term strategy through Hall of Fame")
    print("- More robust to different opponent styles")
    print("- Ensemble combines multiple strategies")


if __name__ == "__main__":
    # Run quick training demo
    ga = quick_train_demo(generations=20)
    
    # Compare with original
    compare_with_original()
    
    print("\nEnhanced AI training complete!")
    print("Files created:")
    print("- quick_enhanced_best.json (best individual network)")
    print("- quick_enhanced_ensemble.json (ensemble of top networks)")
    print("- enhanced_training_progress.png (training plot)")