"""
NBA Player Impact Estimation Model

Hybrid neural architecture combining:
- Player embeddings
- Temporal sequence modeling (LSTM + TCN)
- Possession-level impact prediction

Estimates player marginal impact on team performance per 100 possessions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.models.player_embeddings import PlayerEmbedding, LineupEmbedding, PositionalEncoding
from src.models.temporal_layers import HybridTemporalEncoder


class PlayerImpactModel(nn.Module):
    """
    End-to-end model for estimating player impact on possession outcomes.

    Architecture:
    1. Player Embedding Layer: Encodes players with learned + contextual features
    2. Lineup Interaction Layer: Models player synergies
    3. Temporal Encoder: Processes possession sequences (LSTM + TCN)
    4. Impact Prediction Head: Estimates expected point differential per 100 possessions
    """

    def __init__(self,
                 num_players: int,
                 player_feature_dim: int = 64,
                 event_feature_dim: int = 32,
                 context_feature_dim: int = 16,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_lstm_layers: int = 2,
                 num_tcn_layers: int = 4,
                 dropout: float = 0.2,
                 fusion_method: str = 'concat'):
        """
        Initialize player impact model.

        Args:
            num_players: Total number of unique players
            player_feature_dim: Dimension of player historical features
            event_feature_dim: Dimension of event trajectory features
            context_feature_dim: Dimension of contextual features
            embedding_dim: Dimension of player embeddings
            hidden_dim: Hidden dimension for temporal layers
            num_lstm_layers: Number of LSTM layers
            num_tcn_layers: Number of TCN layers
            dropout: Dropout probability
            fusion_method: Temporal fusion method ('concat', 'add', 'gated')
        """
        super(PlayerImpactModel, self).__init__()

        self.num_players = num_players
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 1. Player embedding module
        self.player_embedding = PlayerEmbedding(
            num_players=num_players,
            embedding_dim=embedding_dim,
            player_feature_dim=player_feature_dim,
            dropout=dropout
        )

        # 2. Lineup interaction module
        self.lineup_embedding = LineupEmbedding(
            player_embedding_dim=embedding_dim,
            lineup_size=5,  # Standard basketball lineup
            num_heads=4,
            dropout=dropout
        )

        # 3. Positional encoding
        self.positional_encoding = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_positions=5,
            dropout=dropout
        )

        # 4. Feature fusion layer
        # Combines: lineup embeddings (5*embedding_dim), event features, context features
        self.lineup_aggregation = nn.Sequential(
            nn.Linear(5 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        fusion_input_dim = hidden_dim + event_feature_dim + context_feature_dim

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # 5. Temporal encoder (LSTM + TCN hybrid)
        self.temporal_encoder = HybridTemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            num_tcn_layers=num_tcn_layers,
            dropout=dropout,
            fusion_method=fusion_method
        )

        # 6. Impact prediction head
        self.impact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)  # Output: expected point differential
        )

        # 7. Per-player impact decomposition (optional, for interpretability)
        self.player_impact_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self,
                player_ids: torch.Tensor,
                player_features: torch.Tensor,
                event_features: torch.Tensor,
                context_features: torch.Tensor,
                return_player_impacts: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            player_ids: Player IDs, shape (batch, 5) for lineup
            player_features: Historical player features, shape (batch, 5, player_feature_dim)
            event_features: Event trajectory features, shape (batch, event_feature_dim)
            context_features: Contextual features, shape (batch, context_feature_dim)
            return_player_impacts: Whether to return individual player impacts

        Returns:
            Dictionary containing:
            - 'impact': Overall lineup impact, shape (batch, 1)
            - 'player_impacts': Individual player impacts, shape (batch, 5) [if requested]
        """
        batch_size = player_ids.size(0)

        # 1. Get player embeddings
        player_embeds = self.player_embedding(player_ids, player_features)  # (batch, 5, embed_dim)

        # 2. Add positional encoding
        player_embeds = self.positional_encoding(player_embeds)  # (batch, 5, embed_dim)

        # 3. Model lineup interactions
        lineup_repr = self.lineup_embedding(player_embeds)  # (batch, 5, embed_dim)

        # 4. Aggregate lineup representation
        lineup_flat = lineup_repr.reshape(batch_size, -1)  # (batch, 5*embed_dim)
        lineup_agg = self.lineup_aggregation(lineup_flat)  # (batch, hidden_dim)

        # 5. Fuse with event and context features
        fused_features = torch.cat([lineup_agg, event_features, context_features], dim=-1)
        fused_features = self.feature_fusion(fused_features)  # (batch, hidden_dim)

        # 6. Add sequence dimension for temporal encoder
        # In practice, this would be a sequence of possessions
        # For single possession prediction, we use sequence length of 1
        fused_features = fused_features.unsqueeze(1)  # (batch, 1, hidden_dim)

        # 7. Temporal encoding
        temporal_repr = self.temporal_encoder(fused_features)  # (batch, 1, hidden_dim)

        # 8. Remove sequence dimension
        temporal_repr = temporal_repr.squeeze(1)  # (batch, hidden_dim)

        # 9. Predict impact
        impact = self.impact_head(temporal_repr)  # (batch, 1)

        outputs = {'impact': impact}

        # 10. Optional: compute per-player impacts
        if return_player_impacts:
            player_impacts = self.player_impact_head(lineup_repr)  # (batch, 5, 1)
            player_impacts = player_impacts.squeeze(-1)  # (batch, 5)
            outputs['player_impacts'] = player_impacts

        return outputs

    def forward_sequence(self,
                        player_ids_seq: torch.Tensor,
                        player_features_seq: torch.Tensor,
                        event_features_seq: torch.Tensor,
                        context_features_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence of possessions.

        Args:
            player_ids_seq: shape (batch, seq_len, 5)
            player_features_seq: shape (batch, seq_len, 5, player_feature_dim)
            event_features_seq: shape (batch, seq_len, event_feature_dim)
            context_features_seq: shape (batch, seq_len, context_feature_dim)

        Returns:
            Predicted impacts for each possession, shape (batch, seq_len, 1)
        """
        batch_size, seq_len = player_ids_seq.size(0), player_ids_seq.size(1)

        # Process each possession in sequence
        sequence_features = []

        for t in range(seq_len):
            # Get features for time step t
            player_ids_t = player_ids_seq[:, t, :]  # (batch, 5)
            player_features_t = player_features_seq[:, t, :, :]  # (batch, 5, player_feature_dim)
            event_features_t = event_features_seq[:, t, :]  # (batch, event_feature_dim)
            context_features_t = context_features_seq[:, t, :]  # (batch, context_feature_dim)

            # Get player embeddings
            player_embeds = self.player_embedding(player_ids_t, player_features_t)
            player_embeds = self.positional_encoding(player_embeds)
            lineup_repr = self.lineup_embedding(player_embeds)

            # Aggregate and fuse
            lineup_flat = lineup_repr.reshape(batch_size, -1)
            lineup_agg = self.lineup_aggregation(lineup_flat)
            fused = torch.cat([lineup_agg, event_features_t, context_features_t], dim=-1)
            fused = self.feature_fusion(fused)

            sequence_features.append(fused)

        # Stack into sequence
        sequence_features = torch.stack(sequence_features, dim=1)  # (batch, seq_len, hidden_dim)

        # Temporal encoding over entire sequence
        temporal_repr = self.temporal_encoder(sequence_features)  # (batch, seq_len, hidden_dim)

        # Predict impact for each time step
        impacts = self.impact_head(temporal_repr)  # (batch, seq_len, 1)

        return impacts


def main():
    """Test player impact model."""
    # Test parameters
    batch_size = 32
    num_players = 500
    player_feature_dim = 64
    event_feature_dim = 32
    context_feature_dim = 16

    # Create sample data
    player_ids = torch.randint(1, num_players, (batch_size, 5))
    player_features = torch.randn(batch_size, 5, player_feature_dim)
    event_features = torch.randn(batch_size, event_feature_dim)
    context_features = torch.randn(batch_size, context_feature_dim)

    # Initialize model
    model = PlayerImpactModel(
        num_players=num_players,
        player_feature_dim=player_feature_dim,
        event_feature_dim=event_feature_dim,
        context_feature_dim=context_feature_dim,
        embedding_dim=128,
        hidden_dim=256
    )

    # Test single possession prediction
    print("Testing single possession prediction...")
    outputs = model(player_ids, player_features, event_features, context_features, return_player_impacts=True)
    print(f"Impact shape: {outputs['impact'].shape}")
    print(f"Player impacts shape: {outputs['player_impacts'].shape}")
    assert outputs['impact'].shape == (batch_size, 1)
    assert outputs['player_impacts'].shape == (batch_size, 5)

    # Test sequence prediction
    print("\nTesting sequence prediction...")
    seq_len = 20
    player_ids_seq = torch.randint(1, num_players, (batch_size, seq_len, 5))
    player_features_seq = torch.randn(batch_size, seq_len, 5, player_feature_dim)
    event_features_seq = torch.randn(batch_size, seq_len, event_feature_dim)
    context_features_seq = torch.randn(batch_size, seq_len, context_feature_dim)

    impacts_seq = model.forward_sequence(
        player_ids_seq, player_features_seq, event_features_seq, context_features_seq
    )
    print(f"Sequence impacts shape: {impacts_seq.shape}")
    assert impacts_seq.shape == (batch_size, seq_len, 1)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nPlayer impact model working correctly!")


if __name__ == "__main__":
    main()
