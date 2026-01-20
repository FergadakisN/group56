# Logging and Hyperparameter Optimization

## Logging

The training pipeline now includes comprehensive logging to both console and file.

### Log Output

- **Console**: INFO level and above (important messages)
- **File**: DEBUG level and above (detailed debug information)
- **Location**: `outputs/{run_name}/training.log`

### Logged Events

Key events logged during training:

- Random seed initialization
- Device resolution (CPU/GPU/MPS)
- Configuration loading from JSON files
- Data loading and class mapping
- Model architecture and parameter count
- Optimizer configuration
- Training progress (epoch, loss, accuracy)
- Checkpoint saving
- Best model updates

### Example Log Output

```txt
INFO - === Starting Training ===
INFO - Run name: resnet_baseline, Output dir: outputs
INFO - Set random seed to 42
INFO - Using device: cuda
DEBUG - Initializing Weights & Biases
INFO - Loading data from data/processed
INFO - Loaded 30 classes: ['anchovy', 'barracuda', 'bellyfish', 'blackfish', 'blenny']...
INFO - Building resnet18 model with 30 classes
INFO - Model created with 11,186,302 trainable parameters
DEBUG - Optimizer setup: LR=0.0003, weight_decay=0.0001
INFO - Starting training for 10 epochs
INFO - Epoch 01/10 | tr_loss: 1.2345 | tr_acc: 0.7234 | va_loss: 1.0987 | va_acc: 0.7890
INFO - New best checkpoint: outputs/resnet_baseline/best.pt (acc=0.7890)
...
INFO - Training process complete.
INFO - Best validation accuracy: 0.8234
INFO - Checkpoints saved to outputs/resnet_baseline
```

## Hyperparameter Optimization (W&B Sweeps)

Use Weights & Biases Sweeps for automated hyperparameter optimization.

### Configuration

The sweep configuration is defined in `configs/sweep.yaml` (Bayesian search). Optionally, `configs/sweep_config.yaml` provides a grid-search variant.

```yaml
program: src/group_56/train.py
method: grid  # grid, random, or bayes
metric:
  name: val/acc
  goal: maximize
parameters:
  arch:
    values: ["resnet18", "resnet34"]
  lr:
    values: [1e-4, 3e-4, 1e-3]
  # ... more parameters
```

### Running a Sweep

#### Method 1: Using the Sweep Agent Script

```bash
# Create and run a new sweep
python src/group_56/sweep_agent.py --config configs/sweep.yaml --count 10

# Join an existing sweep
python src/group_56/sweep_agent.py --sweep-id <SWEEP_ID> --count 5
```

#### Method 2: Manual W&B CLI

```bash
# Create a sweep
wandb sweep configs/sweep.yaml
# Output: Created sweep with ID: group56-fish/1yx2u3v4

# Run sweep agent
wandb agent group56-fish/1yx2u3v4
```

#### Method 3: In Code (Direct)

```python
import wandb
from src.group_56.train import main

sweep_id = wandb.sweep("configs/sweep.yaml", project="group56-fish")
wandb.agent(sweep_id, function=main, count=10)
```

### Viewing Results

View sweep results in the W&B web dashboard:

```txt
https://wandb.ai/<USERNAME>/group56-fish/sweeps/<SWEEP_ID>
```

### Customizing Sweeps

Edit `configs/sweep.yaml` (or `configs/sweep_config.yaml`) to adjust:

- **method**: Grid search (exhaustive), random search, or Bayesian optimization
- **metric**: Which metric to optimize (val/acc, val/loss, etc.)
- **parameters**: Hyperparameter names and values to search over

Example: Bayesian optimization with more parameters

```yaml
method: bayes
metric:
  name: val/acc
  goal: maximize
parameters:
  lr:
    min: 1e-5
    max: 1e-2
  weight_decay:
    min: 1e-6
    max: 1e-3
  batch_size:
    values: [16, 32, 64, 128]
  epochs:
    value: 20
```

## Integration with Training

The training script (`src/group_56/train.py`) is already configured to:

1. Accept hyperparameters via CLI
2. Log all important events to file and console
3. Report metrics to W&B for sweep optimization
4. Save checkpoints and best models

When running under a W&B Sweep, the sweep system automatically:

- Passes sweep parameters to the training script
- Collects metrics logged via `wandb.log()`
- Updates the sweep's optimization objective
- Manages multiple parallel runs

## Best Practices

1. **Define a clear objective**: Always specify which metric to optimize (e.g., validation accuracy)
2. **Set reasonable bounds**: Use sensible min/max for hyperparameters
3. **Limit search space**: Too many combinations can be expensive
4. **Monitor in real-time**: Use the W&B dashboard to watch sweep progress
5. **Use Bayesian optimization** for continuous parameters (more efficient than grid search)
6. **Review logs**: Check `outputs/{run_name}/training.log` for debugging

## Troubleshooting

- **Logs not appearing**: Check that logging is initialized in `_setup_logging()`
- **Sweep not starting**: Ensure W&B is logged in (`wandb login`)
- **Metrics not logged**: Verify `wandb.log()` calls exist in training loop
- **Slow sweeps**: Use random or Bayesian search instead of grid search for large parameter spaces
