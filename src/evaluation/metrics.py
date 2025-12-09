"""
Evaluation Metrics for Player Impact Estimation

Implements metrics for assessing:
- Predictive accuracy
- Temporal stability
- Comparison with traditional metrics
- Model interpretability
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImpactMetrics:
    """
    Comprehensive metrics for evaluating player impact predictions.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.predictions = []
        self.targets = []
        self.player_ids = []
        self.game_ids = []

    def update(self,
               predictions: np.ndarray,
               targets: np.ndarray,
               player_ids: Optional[np.ndarray] = None,
               game_ids: Optional[np.ndarray] = None):
        """
        Update metrics with new predictions.

        Args:
            predictions: Predicted impacts
            targets: Ground truth impacts
            player_ids: Optional player identifiers
            game_ids: Optional game identifiers
        """
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())

        if player_ids is not None:
            self.player_ids.extend(player_ids.flatten())
        if game_ids is not None:
            self.game_ids.extend(game_ids.flatten())

    def compute_basic_metrics(self) -> Dict[str, float]:
        """
        Compute basic regression metrics.

        Returns:
            Dictionary of metrics
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {
            'mse': mean_squared_error(targets, preds),
            'rmse': np.sqrt(mean_squared_error(targets, preds)),
            'mae': mean_absolute_error(targets, preds),
            'r2': r2_score(targets, preds),
        }

        # Correlation metrics
        if len(preds) > 1:
            metrics['pearson_r'], metrics['pearson_p'] = pearsonr(targets, preds)
            metrics['spearman_r'], metrics['spearman_p'] = spearmanr(targets, preds)

        return metrics

    def compute_temporal_stability(self,
                                   window_size: int = 10) -> Dict[str, float]:
        """
        Compute temporal stability of predictions.

        Measures how stable predictions are across consecutive possessions.

        Args:
            window_size: Size of rolling window

        Returns:
            Temporal stability metrics
        """
        if len(self.predictions) < window_size:
            return {'temporal_variance': 0.0, 'temporal_autocorr': 0.0}

        preds = np.array(self.predictions)

        # Rolling variance
        variances = []
        for i in range(len(preds) - window_size + 1):
            window = preds[i:i + window_size]
            variances.append(np.var(window))

        temporal_variance = np.mean(variances)

        # Autocorrelation (lag-1)
        if len(preds) > 1:
            autocorr = np.corrcoef(preds[:-1], preds[1:])[0, 1]
        else:
            autocorr = 0.0

        return {
            'temporal_variance': temporal_variance,
            'temporal_autocorr': autocorr
        }

    def compute_per_player_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Compute metrics per player.

        Returns:
            Dictionary mapping player_id to metrics
        """
        if not self.player_ids:
            logger.warning("No player IDs available for per-player metrics")
            return {}

        player_metrics = {}
        unique_players = set(self.player_ids)

        for player_id in unique_players:
            # Get predictions and targets for this player
            indices = [i for i, pid in enumerate(self.player_ids) if pid == player_id]

            if len(indices) < 2:
                continue

            player_preds = np.array([self.predictions[i] for i in indices])
            player_targets = np.array([self.targets[i] for i in indices])

            # Compute metrics
            player_metrics[player_id] = {
                'mse': mean_squared_error(player_targets, player_preds),
                'mae': mean_absolute_error(player_targets, player_preds),
                'r2': r2_score(player_targets, player_preds) if len(indices) > 1 else 0.0,
                'num_samples': len(indices)
            }

        return player_metrics

    def compute_all_metrics(self) -> Dict[str, any]:
        """
        Compute all available metrics.

        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {}

        # Basic metrics
        metrics['basic'] = self.compute_basic_metrics()

        # Temporal stability
        metrics['temporal'] = self.compute_temporal_stability()

        # Per-player metrics
        per_player = self.compute_per_player_metrics()
        if per_player:
            # Aggregate per-player metrics
            metrics['per_player_avg_mse'] = np.mean([m['mse'] for m in per_player.values()])
            metrics['per_player_avg_mae'] = np.mean([m['mae'] for m in per_player.values()])
            metrics['num_players'] = len(per_player)

        return metrics

    def reset(self):
        """Reset all stored metrics."""
        self.predictions = []
        self.targets = []
        self.player_ids = []
        self.game_ids = []


def compare_with_traditional_metrics(
    model_predictions: np.ndarray,
    traditional_metrics: Dict[str, np.ndarray],
    targets: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compare model predictions with traditional impact metrics.

    Args:
        model_predictions: Neural model predictions
        traditional_metrics: Dict of traditional metric predictions
                           (e.g., {'plus_minus': array, 'rpm': array, 'bpm': array})
        targets: Ground truth values

    Returns:
        Comparison results
    """
    results = {}

    # Evaluate model
    model_mse = mean_squared_error(targets, model_predictions)
    model_mae = mean_absolute_error(targets, model_predictions)
    model_r2 = r2_score(targets, model_predictions)

    results['neural_model'] = {
        'mse': model_mse,
        'mae': model_mae,
        'r2': model_r2
    }

    # Evaluate each traditional metric
    for metric_name, metric_preds in traditional_metrics.items():
        if len(metric_preds) != len(targets):
            logger.warning(f"Skipping {metric_name}: length mismatch")
            continue

        mse = mean_squared_error(targets, metric_preds)
        mae = mean_absolute_error(targets, metric_preds)
        r2 = r2_score(targets, metric_preds)

        results[metric_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }

        # Compute improvement
        results[f'{metric_name}_improvement'] = {
            'mse_reduction': (mse - model_mse) / mse * 100,  # % improvement
            'mae_reduction': (mae - model_mae) / mae * 100,
            'r2_gain': (model_r2 - r2)
        }

    return results


def compute_noise_variance_reduction(
    predictions: np.ndarray,
    targets: np.ndarray,
    baseline_predictions: np.ndarray
) -> float:
    """
    Compute reduction in noise variance compared to baseline.

    Args:
        predictions: Model predictions
        targets: Ground truth
        baseline_predictions: Baseline metric predictions

    Returns:
        Percentage reduction in noise variance
    """
    # Residuals
    model_residuals = targets - predictions
    baseline_residuals = targets - baseline_predictions

    # Variance of residuals (proxy for noise)
    model_noise_var = np.var(model_residuals)
    baseline_noise_var = np.var(baseline_residuals)

    # Reduction
    if baseline_noise_var > 0:
        reduction = (baseline_noise_var - model_noise_var) / baseline_noise_var * 100
    else:
        reduction = 0.0

    return reduction


class CrossValidationMetrics:
    """
    Metrics aggregator for cross-validation across seasons.
    """

    def __init__(self):
        """Initialize cross-validation metrics."""
        self.fold_metrics = []

    def add_fold(self, metrics: Dict[str, float], fold_name: str):
        """
        Add metrics from a single fold.

        Args:
            metrics: Metrics dictionary
            fold_name: Name/identifier for this fold
        """
        self.fold_metrics.append({
            'fold': fold_name,
            'metrics': metrics
        })

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across all folds.

        Returns:
            Dictionary with mean and std for each metric
        """
        if not self.fold_metrics:
            return {}

        # Extract all metric keys
        metric_keys = set()
        for fold_data in self.fold_metrics:
            if 'basic' in fold_data['metrics']:
                metric_keys.update(fold_data['metrics']['basic'].keys())

        # Aggregate
        aggregated = {}
        for key in metric_keys:
            values = []
            for fold_data in self.fold_metrics:
                if 'basic' in fold_data['metrics'] and key in fold_data['metrics']['basic']:
                    values.append(fold_data['metrics']['basic'][key])

            if values:
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        return aggregated

    def print_summary(self):
        """Print summary of cross-validation results."""
        agg = self.aggregate()

        logger.info("=" * 60)
        logger.info("Cross-Validation Results Summary")
        logger.info("=" * 60)

        for metric_name, stats in agg.items():
            logger.info(f"{metric_name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        logger.info("=" * 60)


def main():
    """Test evaluation metrics."""
    # Generate sample data
    n_samples = 1000
    targets = np.random.randn(n_samples) * 10
    predictions = targets + np.random.randn(n_samples) * 2  # Add some noise

    # Test ImpactMetrics
    metrics_tracker = ImpactMetrics()
    metrics_tracker.update(predictions, targets)

    all_metrics = metrics_tracker.compute_all_metrics()
    print("Basic Metrics:")
    for key, value in all_metrics['basic'].items():
        print(f"  {key}: {value:.4f}")

    print("\nTemporal Stability:")
    for key, value in all_metrics['temporal'].items():
        print(f"  {key}: {value:.4f}")

    # Test comparison with traditional metrics
    traditional_metrics = {
        'plus_minus': targets + np.random.randn(n_samples) * 5,
        'rpm': targets + np.random.randn(n_samples) * 4,
    }

    comparison = compare_with_traditional_metrics(predictions, traditional_metrics, targets)
    print("\nComparison with Traditional Metrics:")
    for metric_name, values in comparison.items():
        if 'improvement' not in metric_name:
            print(f"  {metric_name}:")
            for k, v in values.items():
                print(f"    {k}: {v:.4f}")

    print("\nEvaluation metrics working correctly!")


if __name__ == "__main__":
    main()
