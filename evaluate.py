"""
Evaluation Script for NBA Player Impact Model

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/test
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging
import json

from src.models.impact_model import PlayerImpactModel
from src.evaluation.metrics import ImpactMetrics, compare_with_traditional_metrics
from src.utils.config import ExperimentConfig
from src.training.trainer import PossessionDataset
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate NBA Player Impact Model')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not in checkpoint dir)')
    parser.add_argument('--data_dir', type=str, default='data/test',
                       help='Directory containing test data')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')

    return parser.parse_args()


def load_test_data(config: ExperimentConfig, batch_size: int):

    logger.info("Generating dummy test data for demonstration...")

    n_samples = 2000

    # Create dummy features
    player_ids = np.random.randint(1, config.model.num_players, (n_samples, 5))
    player_features = np.random.randn(n_samples, 5, config.model.player_feature_dim)
    event_features = np.random.randn(n_samples, config.model.event_feature_dim)
    context_features = np.random.randn(n_samples, config.model.context_feature_dim)
    targets = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.1, 0.3, 0.2]).astype(np.float32)

    # Create dataset
    test_dataset = PossessionDataset(
        player_ids=player_ids,
        player_features=player_features,
        event_features=event_features,
        context_features=context_features,
        targets=targets
    )

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Loaded {len(test_dataset)} test samples")

    return test_loader


@torch.no_grad()
def evaluate_model(model, test_loader, device):

    model.eval()

    all_predictions = []
    all_targets = []
    all_player_impacts = []

    logger.info("Running evaluation...")

    for batch in test_loader:
        # Move to device
        player_ids = batch['player_ids'].to(device)
        player_features = batch['player_features'].to(device)
        event_features = batch['event_features'].to(device)
        context_features = batch['context_features'].to(device)
        targets = batch['targets'].to(device)

        # Forward pass
        outputs = model(
            player_ids,
            player_features,
            event_features,
            context_features,
            return_player_impacts=True
        )

        # Collect predictions
        all_predictions.append(outputs['impact'].cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_player_impacts.append(outputs['player_impacts'].cpu().numpy())

    # Concatenate
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    player_impacts = np.concatenate(all_player_impacts, axis=0)

    # Compute metrics
    metrics_tracker = ImpactMetrics()
    metrics_tracker.update(predictions, targets)
    metrics = metrics_tracker.compute_all_metrics()

    logger.info("Evaluation complete!")

    return predictions, targets, player_impacts, metrics


def print_results(metrics: dict):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\nBasic Metrics:")
    for key, value in metrics['basic'].items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")

    print("\nTemporal Stability:")
    for key, value in metrics['temporal'].items():
        print(f"  {key:20s}: {value:.4f}")

    if 'num_players' in metrics:
        print(f"\nPer-Player Metrics (across {metrics['num_players']} players):")
        print(f"  {'Average MSE':20s}: {metrics['per_player_avg_mse']:.4f}")
        print(f"  {'Average MAE':20s}: {metrics['per_player_avg_mae']:.4f}")

    print("=" * 70 + "\n")


def save_results(predictions, targets, player_impacts, metrics, output_dir: Path):
    """
    Save evaluation results.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        player_impacts: Individual player impacts
        metrics: Evaluation metrics
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    np.savez(
        output_dir / 'predictions.npz',
        predictions=predictions,
        targets=targets,
        player_impacts=player_impacts
    )
    logger.info(f"Predictions saved to {output_dir / 'predictions.npz'}")

    # Save metrics
    # Convert any non-serializable values
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            serializable_metrics[key] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in value.items()
            }
        else:
            serializable_metrics[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Metrics saved to {output_dir / 'metrics.json'}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = args.config or (checkpoint_dir / 'config.yaml')

    if config_path.exists():
        config = ExperimentConfig.load(str(config_path))
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning("No config file found, using default configuration")
        from src.utils.config import get_default_config
        config = get_default_config()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize model
    logger.info("Initializing model...")
    model = PlayerImpactModel(
        num_players=config.model.num_players,
        player_feature_dim=config.model.player_feature_dim,
        event_feature_dim=config.model.event_feature_dim,
        context_feature_dim=config.model.context_feature_dim,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_lstm_layers=config.model.num_lstm_layers,
        num_tcn_layers=config.model.num_tcn_layers,
        dropout=config.model.dropout,
        fusion_method=config.model.fusion_method
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    logger.info(f"Checkpoint loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")

    # Load test data
    test_loader = load_test_data(config, args.batch_size)

    # Evaluate
    predictions, targets, player_impacts, metrics = evaluate_model(
        model, test_loader, device
    )

    # Print results
    print_results(metrics)

    # Save results
    if args.save_predictions:
        output_dir = Path(args.output_dir)
        save_results(predictions, targets, player_impacts, metrics, output_dir)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
