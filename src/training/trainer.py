"""
Training Pipeline for Player Impact Model

Handles:
- Model training loop
- Validation
- Checkpointing
- Learning rate scheduling
- Early stopping
- Logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import logging
from tqdm import tqdm
import json

from src.models.impact_model import PlayerImpactModel
from src.evaluation.metrics import ImpactMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PossessionDataset(Dataset):
    """
    PyTorch Dataset for possession-level data.
    """

    def __init__(self,
                 player_ids: np.ndarray,
                 player_features: np.ndarray,
                 event_features: np.ndarray,
                 context_features: np.ndarray,
                 targets: np.ndarray):
        """
        Initialize dataset.

        Args:
            player_ids: Array of player IDs, shape (N, 5)
            player_features: Array of player features, shape (N, 5, player_feature_dim)
            event_features: Array of event features, shape (N, event_feature_dim)
            context_features: Array of context features, shape (N, context_feature_dim)
            targets: Target values, shape (N,)
        """
        self.player_ids = torch.LongTensor(player_ids)
        self.player_features = torch.FloatTensor(player_features)
        self.event_features = torch.FloatTensor(event_features)
        self.context_features = torch.FloatTensor(context_features)
        self.targets = torch.FloatTensor(targets).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'player_ids': self.player_ids[idx],
            'player_features': self.player_features[idx],
            'event_features': self.event_features[idx],
            'context_features': self.context_features[idx],
            'targets': self.targets[idx]
        }


class PlayerImpactTrainer:
    """
    Trainer for Player Impact Model.
    """

    def __init__(self,
                 model: PlayerImpactModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 checkpoint_dir: str = 'checkpoints',
                 log_interval: int = 100):
        """
        Initialize trainer.

        Args:
            model: PlayerImpactModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            weight_decay: L2 regularization weight
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging interval (batches)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_interval = log_interval

        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Metrics
        self.metrics_tracker = ImpactMetrics()

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            player_ids = batch['player_ids'].to(self.device)
            player_features = batch['player_features'].to(self.device)
            event_features = batch['event_features'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            targets = batch['targets'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                player_ids,
                player_features,
                event_features,
                context_features
            )

            # Compute loss
            loss = self.criterion(outputs['impact'], targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                logger.info(f'Epoch {epoch}, Batch {batch_idx + 1}/{len(self.train_loader)}, '
                          f'Loss: {avg_loss:.4f}')

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (validation loss, metrics dict)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Reset metrics
        self.metrics_tracker.reset()

        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move to device
            player_ids = batch['player_ids'].to(self.device)
            player_features = batch['player_features'].to(self.device)
            event_features = batch['event_features'].to(self.device)
            context_features = batch['context_features'].to(self.device)
            targets = batch['targets'].to(self.device)

            # Forward pass
            outputs = self.model(
                player_ids,
                player_features,
                event_features,
                context_features
            )

            # Compute loss
            loss = self.criterion(outputs['impact'], targets)
            total_loss += loss.item()
            num_batches += 1

            # Update metrics
            predictions = outputs['impact'].cpu().numpy()
            targets_np = targets.cpu().numpy()
            self.metrics_tracker.update(predictions, targets_np)

        # Compute metrics
        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute_all_metrics()

        logger.info(f'Validation - Epoch {epoch}, Loss: {avg_loss:.4f}')
        logger.info(f'Metrics: {metrics["basic"]}')

        return avg_loss, metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Checkpoint saved to {checkpoint_path}')

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f'Best model saved to {best_path}')

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f'Checkpoint loaded from {checkpoint_path}')

    def train(self,
             num_epochs: int,
             early_stopping_patience: int = 10,
             save_every: int = 5) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_every: Save checkpoint every N epochs

        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate(epoch)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                logger.info(f'Early stopping triggered after {epoch} epochs')
                break

            logger.info(f'Epoch {epoch} complete - Train Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Best Val Loss: {self.best_val_loss:.4f}')

        logger.info("Training complete!")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


def create_dataloaders(
    features_dict: Dict[str, np.ndarray],
    targets: np.ndarray,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        features_dict: Dictionary of feature arrays
        targets: Target array
        batch_size: Batch size
        train_split: Fraction of data for training
        shuffle: Whether to shuffle data

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split data
    n_samples = len(targets)
    n_train = int(n_samples * train_split)

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create datasets
    train_dataset = PossessionDataset(
        player_ids=features_dict['player_ids'][train_indices],
        player_features=features_dict['player_features'][train_indices],
        event_features=features_dict['event_features'][train_indices],
        context_features=features_dict['context_features'][train_indices],
        targets=targets[train_indices]
    )

    val_dataset = PossessionDataset(
        player_ids=features_dict['player_ids'][val_indices],
        player_features=features_dict['player_features'][val_indices],
        event_features=features_dict['event_features'][val_indices],
        context_features=features_dict['context_features'][val_indices],
        targets=targets[val_indices]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    return train_loader, val_loader


def main():
    """Example training script."""
    logger.info("Training pipeline ready")


if __name__ == "__main__":
    main()
