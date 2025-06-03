# Enhanced AI Training Guide 🧠

## Quick Start (Recommended)

### 1. Install Required Dependencies
```bash
pip install numpy numba matplotlib
```

### 2. Run Quick Training Demo (20 generations)
```bash
python3 train_enhanced_quick.py
```
This will:
- Train for 20 generations (15-30 minutes)
- Show progress with plots
- Create `quick_enhanced_best.json` and `quick_enhanced_ensemble.json`

### 3. Convert Trained Model to Web Format
```bash
python3 convert_enhanced_model.py
```
This updates the web model files automatically.

### 4. Test in Browser
```bash
npm run dev
```
Your trained model is now running in the game!

## Full Training Options

### Option A: Full Training (50+ Generations)
```bash
python3 genetic_algorithm_enhanced.py
```
- Takes 1-2 hours
- Better performance
- Creates checkpoints every 10 generations

### Option B: Continue from Checkpoint
```bash
# If training was interrupted, resume from last checkpoint
python3 -c "
from genetic_algorithm_enhanced import AdaptiveGeneticAlgorithm
ga = AdaptiveGeneticAlgorithm()
ga.load_checkpoint('enhanced_checkpoint_gen_20.pkl')
# Continue training...
for i in range(30):  # Train 30 more generations
    ga.evolve_generation()
ga.save_checkpoint('enhanced_checkpoint_final.pkl')
"
```

### Option C: Custom Training Script
```bash
python3 custom_training.py  # (see below)
```

## Performance Testing

### Compare AI Versions
```bash
python3 test_enhanced_ai.py
```
This will:
- Test Enhanced AI vs Original AI
- Show win rates and statistics
- Generate comparison plots
- Analyze feature importance

### Monitor Training Progress
```bash
# Real-time training with progress bar
python3 train_with_progress.py
```

## Advanced Training Configuration

Create `custom_training.py`:
```python
from genetic_algorithm_enhanced import AdaptiveGeneticAlgorithm

# Custom training parameters
ga = AdaptiveGeneticAlgorithm(
    population_size=100,        # Larger = better but slower
    elite_size=10,              # Top performers to keep
    mutation_rate_initial=0.1,  # Starting mutation rate
    crossover_rate=0.7,         # Crossover probability
    tournament_size=5           # Tournament selection size
)

# Train with custom settings
for gen in range(100):  # 100 generations
    ga.evolve_generation()
    
    # Save checkpoint every 10 generations
    if (gen + 1) % 10 == 0:
        ga.save_checkpoint(f'custom_checkpoint_gen_{gen + 1}.pkl')
        print(f"Checkpoint saved at generation {gen + 1}")

# Create final ensemble
ensemble = ga.create_ensemble(top_n=5)
ensemble.save("custom_yaniv_ensemble.json")

# Save best individual
best_idx = ga.fitness_scores.argmax()
ga.population[best_idx].save("custom_yaniv_best.json")
```

## Training Parameters Explained

### Population Size
- **Small (20-50)**: Fast training, may get stuck in local optima
- **Medium (50-100)**: Good balance (recommended)
- **Large (100-200)**: Better exploration, much slower

### Generations
- **Quick test**: 10-20 generations
- **Good results**: 50-100 generations
- **Best performance**: 100+ generations

### Elite Size
- Keep top 5-10% of population
- Ensures good strategies aren't lost

### Mutation Rate
- **High (0.2+)**: More exploration, less exploitation
- **Medium (0.1-0.2)**: Balanced (recommended)
- **Low (0.05-0.1)**: Fine-tuning existing strategies

## Training Tips

### 1. Start Small
```bash
# Quick 10-generation test
python3 -c "
from genetic_algorithm_enhanced import train_enhanced_ai
train_enhanced_ai(generations=10)
"
```

### 2. Use Checkpoints
Always save checkpoints in case training is interrupted:
```python
# In your training script
if (generation + 1) % 10 == 0:
    ga.save_checkpoint(f'checkpoint_gen_{generation + 1}.pkl')
```

### 3. Monitor Progress
Look for these patterns:
- **Best fitness increasing**: Good progress
- **Flat for 10+ generations**: May need more diversity
- **Oscillating**: Normal, algorithm is exploring

### 4. Ensemble Training
Train multiple models and combine them:
```bash
# Train 3 different models
python3 train_enhanced_quick.py  # Model 1
python3 train_enhanced_quick.py  # Model 2  
python3 train_enhanced_quick.py  # Model 3

# Then create ensemble (manually combine best from each)
```

## Hardware Requirements

### Minimum
- **CPU**: 2+ cores
- **RAM**: 4GB
- **Time**: 15-30 minutes (quick training)

### Recommended
- **CPU**: 4+ cores (uses multiprocessing)
- **RAM**: 8GB
- **Time**: 1-2 hours (full training)

### Performance Scaling
- **2 cores**: ~1000 games/minute
- **4 cores**: ~2000 games/minute
- **8 cores**: ~4000 games/minute

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install numpy numba matplotlib
```

### Training is slow
- Reduce population_size to 20-30
- Reduce generations to 20-30
- Use quick training script

### Training gets stuck
- Increase mutation rate: `mutation_rate_initial=0.2`
- Increase population diversity
- Add more stagnation detection

### Memory issues
- Reduce population_size
- Clear unnecessary variables
- Use smaller batch sizes

## Results Interpretation

### Good Training Signs
- Best fitness > 0.7 (70% win rate)
- Steady improvement over generations
- Low stagnation counter

### Warning Signs
- Best fitness < 0.6 (60% win rate)
- No improvement for 20+ generations
- High mutation rates throughout

### Expected Performance
- **Untrained**: ~50% win rate (random)
- **Quick training (20 gen)**: ~65-70% win rate
- **Full training (50+ gen)**: ~70-80% win rate
- **Optimized training**: ~80%+ win rate

## Next Steps After Training

1. **Convert model**: `python3 convert_enhanced_model.py`
2. **Test in browser**: Check performance in actual gameplay
3. **Compare versions**: Use test scripts to validate improvement
4. **Fine-tune**: Adjust parameters and retrain if needed
5. **Share results**: Save your best models!

Happy training! 🚀