# Yaniv AI Training System

This system uses genetic algorithms to evolve neural networks that learn to play Yaniv through self-play.

## How it Works

1. **Population**: Starts with X random neural networks (default: 30)
2. **Tournament**: Each AI plays 10 games against every other AI
3. **Selection**: Top 10 performers survive to next generation
4. **Mutation**: Survivors spawn mutated offspring
5. **Evolution**: Process repeats for many generations

## Files

- `yaniv_neural_network.py` - Simple feedforward neural network
- `genetic_algorithm.py` - Evolution framework
- `yaniv_game_ai.py` - Game logic with legal move validation
- `train_ai.py` - Main training script
- `test_ai.py` - Test trained networks

## Usage

### Training
```bash
python train_ai.py --population 30 --generations 50 --games 10
```

Options:
- `--population`: Number of AI players per generation (default: 30)
- `--generations`: Number of evolution cycles (default: 50)
- `--games`: Games per 1v1 matchup (default: 10)
- `--top-k`: Number of survivors (default: 10)
- `--mutation-rate`: Chance of weight mutation (default: 0.1)
- `--mutation-strength`: Size of mutations (default: 0.1)

### Testing
```bash
python test_ai.py
```

This will:
1. Test a single game between random AIs
2. Test trained AI vs random AI (100 games)

## Performance

The system is CPU-optimized with:
- Small neural networks (64 hidden units)
- Efficient numpy operations
- No deep learning frameworks required

Total games per generation = population × (population-1) × games_per_matchup

Example: 30 AIs × 29 opponents × 10 games = 8,700 games per generation

## Results

- Best networks saved to `saved_networks/`
- Fitness history plotted to `fitness_history.png`
- Networks learn without explicit good/bad move labels