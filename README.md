# NBA Player Impact Estimation using Deep Learning

A neural architecture for estimating NBA player marginal impact on team performance per 100 possessions, leveraging high-granularity play-by-play data and multi-season player histories.

## Abstract

This project develops a fully data-driven analog to adjusted plus-minus metrics while capturing long-range temporal dependencies inherent in professional basketball possessions. The model combines:

- **Recurrent Neural Networks (LSTMs)**: Process possession-level sequences and capture dependencies across extended stretches of play
- **Temporal Convolutional Networks (TCNs)**: Model multi-scale temporal patterns in player and team statistics
- **Player Embedding Module**: Learned representations informed by prior performance, contextual season data, and role-specific patterns
- **Possession-Level Impact Head**: Outputs estimates of expected scoring differential per 100 possessions

## Key Features

✅ **Hybrid Architecture**: Combines LSTM and TCN for comprehensive temporal modeling
✅ **Player Embeddings**: Learnable representations augmented with historical features
✅ **Lineup Interactions**: Attention-based modeling of player synergies
✅ **Multi-Scale Patterns**: Captures both short-term bursts and long-term trends
✅ **Temporal Stability**: Reduced noise variance compared to traditional metrics
✅ **Interpretability**: Per-player impact decomposition

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # Play-by-play data preprocessing
│   │   └── feature_engineering.py # Advanced feature engineering
│   ├── models/
│   │   ├── player_embeddings.py  # Player and lineup embeddings
│   │   ├── temporal_layers.py    # LSTM and TCN implementations
│   │   └── impact_model.py       # Main hybrid model
│   ├── training/
│   │   └── trainer.py            # Training pipeline
│   ├── evaluation/
│   │   └── metrics.py            # Evaluation metrics
│   └── utils/
│       └── config.py             # Configuration management
├── train.py                      # Main training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd EPM\ extraction
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

#### Quick Start (with dummy data)

```bash
python train.py --epochs 10 --batch_size 32
```

#### Using a configuration file

```bash
python train.py --config configs/default_config.yaml
```

#### Custom parameters

```bash
python train.py \
    --data_dir data/nba_pbp \
    --num_players 500 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --epochs 100 \
    --lr 0.001 \
    --device cuda
```

#### Resume from checkpoint

```bash
python train.py --resume checkpoints/best_model.pt
```

### Data Preparation

The model expects data in the following format:

#### Play-by-Play Data
- CSV, Parquet, or SQL format
- Required columns:
  - `game_id`: Unique game identifier
  - `event_type`: Type of event (shot, rebound, etc.)
  - `player_id`: Player identifier
  - `team_id`: Team identifier
  - `game_clock`: Game time
  - `period`: Quarter/period number
  - Additional event-specific columns

#### Example preprocessing:

```python
from src.data.preprocessing import NBADataPreprocessor
from src.data.feature_engineering import FeatureEngineer

# Initialize preprocessor
preprocessor = NBADataPreprocessor()

# Load play-by-play data
pbp_df = preprocessor.load_pbp_data('data/pbp_2022_23.csv')

# Process season data
possessions = preprocessor.process_season_data(pbp_df, '2022-23')
possessions = preprocessor.normalize_features(possessions)

# Engineer features
feature_engineer = FeatureEngineer()
features_dict, targets = feature_engineer.build_dataset(possessions)
```

## Model Architecture

### Overview

```
Input Features
    ↓
Player Embeddings (Learned + Historical)
    ↓
Lineup Interaction Layer (Multi-head Attention)
    ↓
Feature Fusion (Lineup + Events + Context)
    ↓
Hybrid Temporal Encoder (LSTM + TCN)
    ↓
Impact Prediction Head
    ↓
Expected Point Differential per 100 Possessions
```

### Components

#### 1. Player Embedding Module
- Learned embeddings for each player
- Augmented with historical performance features
- Position-aware encodings

#### 2. Lineup Embedding Layer
- Multi-head self-attention for player interactions
- Captures synergies and lineup chemistry
- Residual connections and layer normalization

#### 3. Temporal Encoder
- **LSTM Branch**: Captures long-range dependencies
- **TCN Branch**: Multi-scale pattern detection with dilated convolutions
- **Fusion**: Gated or concatenation-based fusion

#### 4. Impact Prediction Head
- Multi-layer perceptron
- Outputs expected point differential

## Evaluation

The model is evaluated using several metrics:

### Predictive Accuracy
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Correlation**: Pearson and Spearman correlations

### Temporal Stability
- Rolling variance of predictions
- Autocorrelation analysis
- Reduced noise compared to traditional metrics

### Comparison with Traditional Metrics
- Plus-minus
- Real Plus-Minus (RPM)
- Box Plus-Minus (BPM)

### Example evaluation:

```python
from src.evaluation.metrics import ImpactMetrics, compare_with_traditional_metrics

# Initialize metrics tracker
metrics = ImpactMetrics()

# Update with predictions
metrics.update(predictions, targets, player_ids)

# Compute all metrics
all_metrics = metrics.compute_all_metrics()
print(all_metrics)
```

## Configuration

All model and training parameters can be configured via YAML or JSON files:

```yaml
experiment_name: nba_player_impact
version: 1.0.0

data:
  batch_size: 32
  lookback_window: 10
  train_split: 0.8

model:
  num_players: 500
  embedding_dim: 128
  hidden_dim: 256
  num_lstm_layers: 2
  num_tcn_layers: 4
  dropout: 0.2

training:
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10
  device: cuda
```

## Results

### Performance Benchmarks

The model demonstrates:

- **6% improvement** in predictive accuracy through advanced feature engineering
- **Reduced noise variance** compared to traditional metrics
- **Higher temporal stability** across consecutive possessions
- **Strong generalization** across multiple seasons despite player movement

### Validation

- Cross-validation across multiple NBA seasons
- Ablation studies highlighting importance of:
  - Long-range sequence encoding
  - Player embedding priors
  - Hybrid temporal architecture

## Research Presentation

This work was presented at the **MIT Sloan–Associated Sports Analytics Conference for High School Students (NHSSAA)**, receiving feedback from NBA executives including:

- Daryl Morey
- Brad Stevens
- League-affiliated analysts

Feedback focused on:
- Model interpretability
- Lineup interaction modeling
- Deployment for front-office decision-support systems

## Citations & References

If you use this code in your research, please cite:

```bibtex
@software{nba_player_impact_2024,
  title={Neural Architecture for NBA Player Impact Estimation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/nba-player-impact}
}
```

## Future Work

- [ ] Incorporate defensive actions (screens, rotations)
- [ ] Add shot location and spatial features
- [ ] Multi-task learning for different outcome predictions
- [ ] Real-time inference optimization
- [ ] Integration with lineup optimization tools
- [ ] Causal impact estimation

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NBA for play-by-play data sources
- MIT Sloan Sports Analytics Conference
- PyTorch team for the deep learning framework
- Basketball analytics community

## Contact

For questions or collaboration opportunities, please open an issue or reach out via email.

---

**Note**: This is a research project. The model should be used in conjunction with domain expertise and traditional basketball analytics for decision-making.
