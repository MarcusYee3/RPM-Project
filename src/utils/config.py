"""
Configuration Management for Player Impact Model

Centralized configuration for all model parameters, training settings,
and data processing options.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Data paths
    data_dir: str = "data"
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"

    # Preprocessing
    min_possession_events: int = 1
    max_possession_duration: float = 35.0
    normalize_by_pace: bool = True

    # Feature engineering
    lookback_window: int = 10
    interaction_depth: int = 2

    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Player embeddings
    num_players: int = 500
    embedding_dim: int = 128
    player_feature_dim: int = 64

    # Event features
    event_feature_dim: int = 32
    context_feature_dim: int = 16

    # Temporal encoder
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    num_tcn_layers: int = 4
    fusion_method: str = "concat"  # 'concat', 'add', 'gated'

    # Regularization
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100

    # Learning rate scheduling
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Early stopping
    early_stopping_patience: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    log_interval: int = 100

    # Device
    device: str = "cuda"  # or 'cpu'

    # Reproducibility
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Metrics
    compute_per_player: bool = True
    compute_temporal_stability: bool = True
    temporal_window_size: int = 10

    # Comparison
    traditional_metrics: List[str] = field(default_factory=lambda: ["plus_minus", "rpm", "bpm"])

    # Output
    results_dir: str = "results"
    save_predictions: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment metadata
    experiment_name: str = "nba_player_impact"
    description: str = "Neural architecture for NBA player impact estimation"
    version: str = "1.0.0"

    def save(self, path: str):
        """
        Save configuration to file.

        Args:
            path: Path to save config (YAML or JSON)
        """
        path = Path(path)
        config_dict = self.to_dict()

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """
        Load configuration from file.

        Args:
            path: Path to config file (YAML or JSON)

        Returns:
            ExperimentConfig instance
        """
        path = Path(path)

        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Reconstruct nested config
        config = cls()
        config.data = DataConfig(**config_dict.get('data', {}))
        config.model = ModelConfig(**config_dict.get('model', {}))
        config.training = TrainingConfig(**config_dict.get('training', {}))
        config.evaluation = EvaluationConfig(**config_dict.get('evaluation', {}))

        # Update top-level fields
        for key in ['experiment_name', 'description', 'version']:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        print(f"Configuration loaded from {path}")
        return config

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'version': self.version,
            'data': {
                k: v for k, v in self.data.__dict__.items()
            },
            'model': {
                k: v for k, v in self.model.__dict__.items()
            },
            'training': {
                k: v for k, v in self.training.__dict__.items()
            },
            'evaluation': {
                k: v for k, v in self.evaluation.__dict__.items()
            }
        }

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print(f"Experiment: {self.experiment_name}")
        print(f"Version: {self.version}")
        print(f"Description: {self.description}")
        print("=" * 70)

        print("\nData Configuration:")
        for key, value in self.data.__dict__.items():
            print(f"  {key:30s}: {value}")

        print("\nModel Configuration:")
        for key, value in self.model.__dict__.items():
            print(f"  {key:30s}: {value}")

        print("\nTraining Configuration:")
        for key, value in self.training.__dict__.items():
            print(f"  {key:30s}: {value}")

        print("\nEvaluation Configuration:")
        for key, value in self.evaluation.__dict__.items():
            print(f"  {key:30s}: {value}")

        print("=" * 70)


def get_default_config() -> ExperimentConfig:
    """
    Get default experiment configuration.

    Returns:
        ExperimentConfig with default settings
    """
    return ExperimentConfig()


def main():
    """Test configuration management."""
    # Create default config
    config = get_default_config()

    # Print summary
    config.print_summary()

    # Save config
    config.save("configs/default_config.yaml")

    # Load config
    loaded_config = ExperimentConfig.load("configs/default_config.yaml")

    print("\nConfiguration management working correctly!")


if __name__ == "__main__":
    main()
