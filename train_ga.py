#!/usr/bin/env python3
"""
Yaniv GA Training Script

Usage:
    python train_ga.py                    # Quick training (10 gen, 32 pop)
    python train_ga.py --full             # Full training (100 gen, 128 pop)
    python train_ga.py --resume checkpoint_dir  # Resume from checkpoint

Example:
    python train_ga.py --generations 50 --population 64 --workers 4
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python_yaniv.ga import GeneticAlgorithm, GAConfig, train
from python_yaniv.export import export_for_typescript


def main():
    parser = argparse.ArgumentParser(
        description="Train Yaniv AI using Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test run (5 generations, small population)
    python train_ga.py --quick

    # Standard training (50 generations)
    python train_ga.py --generations 50

    # Full training with parallel evaluation
    python train_ga.py --full --workers 8

    # Resume from checkpoint
    python train_ga.py --resume ga_checkpoints/gen_0020

    # Export trained model to TypeScript
    python train_ga.py --export ga_checkpoints/best_final.json
        """
    )

    # Training modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick", action="store_true",
        help="Quick test run (5 generations, 16 population)"
    )
    mode_group.add_argument(
        "--full", action="store_true",
        help="Full training (100 generations, 128 population)"
    )

    # Training parameters
    parser.add_argument(
        "--generations", "-g", type=int, default=50,
        help="Number of generations (default: 50)"
    )
    parser.add_argument(
        "--population", "-p", type=int, default=64,
        help="Population size (default: 64)"
    )
    parser.add_argument(
        "--elite", "-e", type=int, default=8,
        help="Elite count (default: 8)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Use sequential (non-parallel) evaluation"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir", "-c", type=str, default="ga_checkpoints",
        help="Directory for checkpoints (default: ga_checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=5,
        help="Checkpoint frequency in generations (default: 5)"
    )

    # Resume / Export
    parser.add_argument(
        "--resume", "-r", type=str, default=None,
        help="Resume from checkpoint directory"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export genome to TypeScript (provide genome JSON path)"
    )
    parser.add_argument(
        "--export-dir", type=str, default="src/game",
        help="TypeScript export directory (default: src/game)"
    )

    # Other
    parser.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Handle export mode
    if args.export:
        print(f"Exporting model from {args.export} to {args.export_dir}")
        export_for_typescript(args.export, args.export_dir)
        return

    # Set up config based on mode
    if args.quick:
        generations = 5
        population = 16
        elite = 4
    elif args.full:
        generations = 100
        population = 128
        elite = 16
    else:
        generations = args.generations
        population = args.population
        elite = args.elite

    print("=" * 60)
    print("Yaniv GA Training")
    print("=" * 60)
    print(f"Generations: {generations}")
    print(f"Population: {population}")
    print(f"Elite count: {elite}")
    print(f"Workers: {args.workers}")
    print(f"Parallel: {not args.sequential}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 60)

    # Create config
    config = GAConfig(
        num_generations=generations,
        population_size=population,
        elite_count=elite,
        num_workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_freq,
        random_seed=args.seed
    )

    # Create and run GA
    ga = GeneticAlgorithm(config)

    if args.resume:
        ga.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume}")
    else:
        ga.initialize()

    best = ga.run(parallel=not args.sequential)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best genome: {best}")
    print(f"Fitness: {best.fitness:.4f}")
    print(f"Win rate: {best.wins}/{best.games_played}")
    print(f"\nTo export to TypeScript:")
    print(f"  python train_ga.py --export {args.checkpoint_dir}/best_final.json")


if __name__ == "__main__":
    main()
