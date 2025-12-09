# Quick Start Guide

Get up and running with the NBA Player Impact Model in 5 minutes.

## 1. Installation

```bash
# Clone the repository (if from git)
# git clone <your-repo-url>
# cd EPM\ extraction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Run Examples

See how the model works with example code:

```bash
python example.py
```

This will demonstrate:
- Data preprocessing
- Feature engineering
- Model initialization
- Making predictions
- Training setup
- Evaluation

## 3. Train with Dummy Data

Train the model with generated dummy data to test the pipeline:

```bash
# Quick training run (10 epochs)
python train.py --epochs 10 --batch_size 32

# The model will:
# - Generate dummy possession data
# - Train for 10 epochs
# - Save checkpoints to checkpoints/
# - Log training progress
```

Expected output:
```
Epoch 1/10 - Train Loss: 1.2345, Val Loss: 1.3456
Epoch 2/10 - Train Loss: 1.1234, Val Loss: 1.2345
...
Training complete!
Best validation loss: 1.1000
```

## 4. Evaluate the Model

After training, evaluate the model:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --save_predictions
```

Results will be saved to `results/`:
- `predictions.npz`: Predicted impacts
- `metrics.json`: Evaluation metrics

## 5. Using Your Own Data

### Data Format

Your play-by-play data should include:

```python
# Required columns:
- game_id: Unique game identifier
- event_type: Type of event (shot_made, shot_missed, rebound, etc.)
- player_id: Player identifier
- team_id: Team identifier
- game_clock: Game time (seconds)
- period: Quarter/period number

# Optional but recommended:
- score_differential: Current score difference
- home_lineup: List of home player IDs
- away_lineup: List of away player IDs
- shot_value: Points for made shots (2 or 3)
```

### Preprocessing Your Data

```python
from src.data.preprocessing import NBADataPreprocessor

# Initialize preprocessor
preprocessor = NBADataPreprocessor()

# Load your data
pbp_df = preprocessor.load_pbp_data('path/to/your/data.csv')

# Process into possessions
possessions = preprocessor.process_season_data(pbp_df, season='2022-23')
possessions = preprocessor.normalize_features(possessions)
```

### Feature Engineering

```python
from src.data.feature_engineering import FeatureEngineer

# Initialize feature engineer
feature_engineer = FeatureEngineer(lookback_window=10)

# Build dataset
features_dict, targets = feature_engineer.build_dataset(possessions)
```

### Train on Your Data

```python
from src.training.trainer import create_dataloaders, PlayerImpactTrainer
from src.models.impact_model import PlayerImpactModel

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    features_dict=features_dict,
    targets=targets,
    batch_size=32
)

# Initialize model
model = PlayerImpactModel(
    num_players=500,  # Adjust based on your data
    embedding_dim=128,
    hidden_dim=256
)

# Train
trainer = PlayerImpactTrainer(model, train_loader, val_loader)
trainer.train(num_epochs=100)
```

## 6. Configuration

Customize model parameters using config files:

```bash
# Edit configs/default_config.yaml
vim configs/default_config.yaml

# Train with custom config
python train.py --config configs/default_config.yaml
```

Key parameters to tune:
```yaml
model:
  embedding_dim: 128      # Player embedding size
  hidden_dim: 256         # Temporal encoder hidden size
  num_lstm_layers: 2      # LSTM depth
  num_tcn_layers: 4       # TCN depth
  dropout: 0.2            # Dropout rate

training:
  learning_rate: 0.001    # Learning rate
  batch_size: 32          # Batch size
  num_epochs: 100         # Number of epochs
```

## 7. Making Predictions

```python
import torch
from src.models.impact_model import PlayerImpactModel

# Load trained model
model = PlayerImpactModel(...)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input data
player_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Lineup
player_features = torch.randn(1, 5, 64)       # Historical features
event_features = torch.randn(1, 32)           # Event trajectory
context_features = torch.randn(1, 16)         # Context

# Predict
with torch.no_grad():
    outputs = model(
        player_ids,
        player_features,
        event_features,
        context_features,
        return_player_impacts=True
    )

lineup_impact = outputs['impact'].item()
player_impacts = outputs['player_impacts'].numpy()

print(f"Lineup Impact: {lineup_impact:.2f} points per 100 possessions")
print(f"Player Impacts: {player_impacts}")
```

## Common Issues

### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch_size 16

# Or use CPU
python train.py --device cpu
```

### Import Errors
```bash
# Make sure you're in the project root directory
cd EPM\ extraction

# Reinstall dependencies
pip install -r requirements.txt
```

### Data Format Issues
- Ensure your CSV has the required columns
- Check for missing values
- Verify player IDs are integers
- Confirm timestamps are in seconds

## Next Steps

1. **Explore the Code**: Check out `src/` directory for implementation details
2. **Customize Features**: Modify `src/data/feature_engineering.py` to add domain-specific features
3. **Tune Hyperparameters**: Experiment with different model architectures
4. **Add New Metrics**: Extend `src/evaluation/metrics.py` with custom evaluation metrics
5. **Deploy**: Use trained model for real-time impact prediction

## Resources

- **Full Documentation**: See [README.md](README.md)
- **Model Architecture**: See research abstract in documentation
- **Examples**: Run `python example.py` for usage examples

## Support

For questions or issues:
1. Check the [README.md](README.md) for detailed documentation
2. Review the example code in `example.py`
3. Open an issue on GitHub (if applicable)

---

**Happy modeling!** 🏀📊
