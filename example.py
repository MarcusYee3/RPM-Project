"""
Example Usage of NBA Player Impact Model


This is a simplified example for demonstration purposes.
"""

import torch
import numpy as np
from pathlib import Path

# Import model components
from src.models.impact_model import PlayerImpactModel
from src.data.preprocessing import NBADataPreprocessor, PossessionSequence
from src.data.feature_engineering import FeatureEngineer
from src.training.trainer import PlayerImpactTrainer, PossessionDataset, create_dataloaders
from src.evaluation.metrics import ImpactMetrics
from src.utils.config import ExperimentConfig, get_default_config


def example_preprocessing():
    """Example: Data Preprocessing"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Data Preprocessing")
    print("=" * 70)

    # Initialize preprocessor
    preprocessor = NBADataPreprocessor(
        min_possession_events=1,
        max_possession_duration=35.0,
        normalize_by_pace=True
    )

    # In practice, you would load actual play-by-play data:
    # pbp_df = preprocessor.load_pbp_data('data/pbp_2022_23.csv')
    # possessions = preprocessor.process_season_data(pbp_df, '2022-23')
    # possessions = preprocessor.normalize_features(possessions)

    print("✓ Preprocessor initialized")
    print("  - Ready to process play-by-play data")
    print("  - Handles possession segmentation")
    print("  - Extracts temporal features")


def example_feature_engineering():
    """Example: Feature Engineering"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Feature Engineering")
    print("=" * 70)

    # Initialize feature engineer
    feature_engineer = FeatureEngineer(
        lookback_window=10,
        interaction_depth=2
    )

    # Create a sample possession
    sample_possession = PossessionSequence(
        possession_id="game_001_poss_1",
        game_id="game_001",
        season="2022-23",
        lineup_home=[1, 2, 3, 4, 5],
        lineup_away=[6, 7, 8, 9, 10],
        events=[{'event_type': 'shot_made', 'player_id': 1}],
        score_differential=5.0,
        possession_result=2.0,
        timestamp_start=720.0,
        timestamp_end=715.0,
        pace_factor=100.0,
        fatigue_indicator=0.5
    )

    # Extract features for this possession
    # features = feature_engineer.build_possession_features(
    #     sample_possession,
    #     all_possessions=[sample_possession],
    #     target_team='home'
    # )

    print("✓ Feature engineer initialized")
    print("  - Extracts player historical features")
    print("  - Computes lineup interaction features")
    print("  - Encodes event trajectories")
    print("  - Generates contextual features")


def example_model_initialization():
    """Example: Model Initialization"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Model Initialization")
    print("=" * 70)

    # Get default configuration
    config = get_default_config()

    # Initialize model
    model = PlayerImpactModel(
        num_players=config.model.num_players,
        player_feature_dim=config.model.player_feature_dim,
        event_feature_dim=config.model.event_feature_dim,
        context_feature_dim=config.model.context_feature_dim,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_lstm_layers=config.model.num_lstm_layers,
        num_tcn_layers=config.model.num_tcn_layers,
        dropout=config.model.dropout
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✓ Model initialized successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model architecture: Hybrid LSTM + TCN")


def example_inference():
    """Example: Making Predictions"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Making Predictions")
    print("=" * 70)

    # Initialize model
    config = get_default_config()
    model = PlayerImpactModel(
        num_players=config.model.num_players,
        player_feature_dim=config.model.player_feature_dim,
        event_feature_dim=config.model.event_feature_dim,
        context_feature_dim=config.model.context_feature_dim,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim
    )
    model.eval()

    # Create sample input data
    batch_size = 1
    player_ids = torch.randint(1, config.model.num_players, (batch_size, 5))
    player_features = torch.randn(batch_size, 5, config.model.player_feature_dim)
    event_features = torch.randn(batch_size, config.model.event_feature_dim)
    context_features = torch.randn(batch_size, config.model.context_feature_dim)

    # Make prediction
    with torch.no_grad():
        outputs = model(
            player_ids,
            player_features,
            event_features,
            context_features,
            return_player_impacts=True
        )

    # Extract predictions
    lineup_impact = outputs['impact'].item()
    player_impacts = outputs['player_impacts'].squeeze().numpy()

    print("✓ Prediction complete")
    print(f"  - Lineup impact: {lineup_impact:.3f} points per 100 possessions")
    print(f"  - Individual player impacts: {player_impacts}")
    print(f"  - Sum of player impacts: {player_impacts.sum():.3f}")


def example_training_loop():
    """Example: Training the Model"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Training Loop")
    print("=" * 70)

    # Get configuration
    config = get_default_config()

    # Generate dummy data
    n_samples = 1000
    features_dict = {
        'player_ids': np.random.randint(1, config.model.num_players, (n_samples, 5)),
        'player_features': np.random.randn(n_samples, 5, config.model.player_feature_dim),
        'event_features': np.random.randn(n_samples, config.model.event_feature_dim),
        'context_features': np.random.randn(n_samples, config.model.context_feature_dim),
    }
    targets = np.random.choice([0, 1, 2, 3], size=n_samples).astype(np.float32)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        features_dict=features_dict,
        targets=targets,
        batch_size=32,
        train_split=0.8
    )

    # Initialize model
    model = PlayerImpactModel(
        num_players=config.model.num_players,
        player_feature_dim=config.model.player_feature_dim,
        event_feature_dim=config.model.event_feature_dim,
        context_feature_dim=config.model.context_feature_dim,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim
    )

    print("✓ Training setup complete")
    print(f"  - Training samples: {len(train_loader.dataset)}")
    print(f"  - Validation samples: {len(val_loader.dataset)}")
    print(f"  - Batch size: 32")
    print("  - Ready to train with: trainer.train(num_epochs=100)")


def example_evaluation():
    """Example: Model Evaluation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Model Evaluation")
    print("=" * 70)

    # Generate dummy predictions and targets
    n_samples = 500
    predictions = np.random.randn(n_samples) * 10
    targets = predictions + np.random.randn(n_samples) * 2  # Add noise

    # Initialize metrics tracker
    metrics = ImpactMetrics()
    metrics.update(predictions, targets)

    # Compute metrics
    all_metrics = metrics.compute_all_metrics()

    print("✓ Evaluation complete")
    print("  Basic Metrics:")
    for key, value in all_metrics['basic'].items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.4f}")

    print("  Temporal Stability:")
    for key, value in all_metrics['temporal'].items():
        print(f"    - {key}: {value:.4f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("NBA PLAYER IMPACT MODEL - USAGE EXAMPLES")
    print("=" * 70)

    # Run examples
    example_preprocessing()
    example_feature_engineering()
    example_model_initialization()
    example_inference()
    example_training_loop()
    example_evaluation()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
To train your own model:

1. Prepare your data:
   - Format play-by-play data as CSV/Parquet
   - Run preprocessing pipeline
   - Generate features

2. Configure the model:
   - Edit configs/default_config.yaml
   - Adjust hyperparameters as needed

3. Train the model:
   python train.py --config configs/default_config.yaml --epochs 100

4. Evaluate the model:
   python evaluate.py --checkpoint checkpoints/best_model.pt

5. Make predictions:
   - Load trained model
   - Pass in new possession data
   - Get impact estimates

For more details, see README.md
    """)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
