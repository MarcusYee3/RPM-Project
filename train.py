"""
Main Training Script for NBA Player Impact Model

Usage:
    python train.py --config configs/default_config.yaml
    python train.py --data_dir data/nba_pbp --epochs 100
"""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import logging

from src.models.impact_model import PlayerImpactModel
from src.training.trainer import PlayerImpactTrainer, create_dataloaders
from src.utils.config import ExperimentConfig, get_default_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train NBA Player Impact Model')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (YAML or JSON)')

    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing training data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')

    # Model
    parser.add_argument('--num_players', type=int, default=500,
                       help='Total number of unique players')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Dimension of player embeddings')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for temporal layers')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval (batches)')

    return parser.parse_args()


def load_dummy_data(config: ExperimentConfig):
    """
    Load dummy data for demonstration.

    """
    logger.info("Generating dummy data for demonstration...")

    n_samples = 10000  # Number of possessions

    # Create dummy features
    features_dict = {
        'player_ids': np.random.randint(1, config.model.num_players, (n_samples, 5)),
        'player_features': np.random.randn(n_samples, 5, config.model.player_feature_dim),
        'event_features': np.random.randn(n_samples, config.model.event_feature_dim),
        'context_features': np.random.randn(n_samples, config.model.context_feature_dim),
    }

    # Create dummy targets (points per possession, 0-3)
    targets = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.1, 0.3, 0.2]).astype(np.float32)

    logger.info(f"Generated {n_samples} dummy possessions")

    return features_dict, targets


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = get_default_config()

        # Override with command line arguments
        config.data.batch_size = args.batch_size
        config.model.num_players = args.num_players
        config.model.embedding_dim = args.embedding_dim
        config.model.hidden_dim = args.hidden_dim
        config.training.num_epochs = args.epochs
        config.training.learning_rate = args.lr
        config.training.device = args.device
        config.training.seed = args.seed
        config.training.checkpoint_dir = args.checkpoint_dir
        config.training.log_interval = args.log_interval

    # Print configuration
    config.print_summary()

    # Set random seed
    set_seed(config.training.seed)

    # Create checkpoint directory
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config.save(Path(config.training.checkpoint_dir) / 'config.yaml')

    # Load data
    # NOTE: Replace load_dummy_data with actual data loading
    features_dict, targets = load_dummy_data(config)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        features_dict=features_dict,
        targets=targets,
        batch_size=config.data.batch_size,
        train_split=config.data.train_split,
        shuffle=True
    )

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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized with {total_params:,} total parameters "
               f"({trainable_params:,} trainable)")

    # Initialize trainer
    trainer = PlayerImpactTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=config.training.device,
        checkpoint_dir=config.training.checkpoint_dir,
        log_interval=config.training.log_interval
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    history = trainer.train(
        num_epochs=config.training.num_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_every=config.training.save_every
    )

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

    # Save final model
    final_checkpoint_path = Path(config.training.checkpoint_dir) / 'final_model.pt'
    trainer.save_checkpoint(config.training.num_epochs, is_best=False)

    logger.info(f"Final model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
