"""
Player Embedding Module

Learns dense representations for NBA players informed by:
- Historical performance data
- Role-specific patterns
- Contextual season information
- Positional priors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class PlayerEmbedding(nn.Module):
    """
    Learnable player embeddings with contextual augmentation.

    Each player is represented by:
    1. A learned embedding vector
    2. Contextual features (season stats, role indicators)
    3. Position encoding
    """

    def __init__(self,
                 num_players: int,
                 embedding_dim: int = 128,
                 player_feature_dim: int = 64,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        """
        Initialize player embedding module.

        Args:
            num_players: Total number of unique players
            embedding_dim: Dimension of learned embeddings
            player_feature_dim: Dimension of input player features
            dropout: Dropout probability
            use_layer_norm: Whether to apply layer normalization
        """
        super(PlayerEmbedding, self).__init__()

        self.num_players = num_players
        self.embedding_dim = embedding_dim
        self.player_feature_dim = player_feature_dim

        # Learned embeddings for each player
        self.player_embeddings = nn.Embedding(
            num_embeddings=num_players,
            embedding_dim=embedding_dim,
            padding_idx=0  # Reserve 0 for unknown/padding
        )

        # Feature projection network
        # Projects historical features to same dimension as embeddings
        self.feature_projection = nn.Sequential(
            nn.Linear(player_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Fusion layer to combine embeddings and features
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim) if use_layer_norm else None

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.xavier_uniform_(self.player_embeddings.weight)
        # Set padding embedding to zero
        with torch.no_grad():
            self.player_embeddings.weight[0] = 0

    def forward(self,
                player_ids: torch.Tensor,
                player_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to get player representations.

        Args:
            player_ids: Tensor of player IDs, shape (batch_size, num_players)
            player_features: Optional historical features, shape (batch_size, num_players, feature_dim)

        Returns:
            Player embeddings, shape (batch_size, num_players, embedding_dim)
        """
        # Get learned embeddings
        embeddings = self.player_embeddings(player_ids)  # (batch, num_players, embed_dim)

        if player_features is not None:
            # Project features
            projected_features = self.feature_projection(player_features)  # (batch, num_players, embed_dim)

            # Concatenate and fuse
            combined = torch.cat([embeddings, projected_features], dim=-1)  # (batch, num_players, 2*embed_dim)
            embeddings = self.fusion(combined)  # (batch, num_players, embed_dim)

        # Layer normalization
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)

        return embeddings


class LineupEmbedding(nn.Module):
    """
    Lineup-level embedding that captures player interactions.

    Uses attention mechanism to model synergies between players.
    """

    def __init__(self,
                 player_embedding_dim: int = 128,
                 lineup_size: int = 5,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize lineup embedding module.

        Args:
            player_embedding_dim: Dimension of player embeddings
            lineup_size: Number of players in lineup (typically 5)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(LineupEmbedding, self).__init__()

        self.player_embedding_dim = player_embedding_dim
        self.lineup_size = lineup_size

        # Multi-head self-attention for player interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=player_embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(player_embedding_dim)
        self.norm2 = nn.LayerNorm(player_embedding_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(player_embedding_dim, player_embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(player_embedding_dim * 4, player_embedding_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, player_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get lineup representation.

        Args:
            player_embeddings: Tensor of player embeddings, shape (batch, lineup_size, embed_dim)

        Returns:
            Lineup representation, shape (batch, lineup_size, embed_dim)
        """
        # Self-attention over players in lineup
        attn_output, _ = self.self_attention(
            player_embeddings,
            player_embeddings,
            player_embeddings
        )

        # Residual connection + normalization
        x = self.norm1(player_embeddings + self.dropout(attn_output))

        # Feed-forward network
        ffn_output = self.ffn(x)

        # Residual connection + normalization
        lineup_repr = self.norm2(x + self.dropout(ffn_output))

        return lineup_repr


class PositionalEncoding(nn.Module):
    """
    Positional encoding for player positions (PG, SG, SF, PF, C).
    """

    def __init__(self,
                 embedding_dim: int,
                 max_positions: int = 5,
                 dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            embedding_dim: Dimension of embeddings
            max_positions: Maximum number of positions (5 for basketball)
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-np.log(10000.0) / embedding_dim))

        pe = torch.zeros(max_positions, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor, shape (batch, seq_len, embed_dim)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


def main():
    """Test player embedding modules."""
    # Test parameters
    batch_size = 32
    num_players = 500
    lineup_size = 5
    embedding_dim = 128
    player_feature_dim = 64

    # Create player IDs and features
    player_ids = torch.randint(0, num_players, (batch_size, lineup_size))
    player_features = torch.randn(batch_size, lineup_size, player_feature_dim)

    # Test PlayerEmbedding
    player_emb = PlayerEmbedding(num_players, embedding_dim, player_feature_dim)
    embeddings = player_emb(player_ids, player_features)
    print(f"Player embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (batch_size, lineup_size, embedding_dim)

    # Test LineupEmbedding
    lineup_emb = LineupEmbedding(embedding_dim, lineup_size)
    lineup_repr = lineup_emb(embeddings)
    print(f"Lineup representation shape: {lineup_repr.shape}")
    assert lineup_repr.shape == (batch_size, lineup_size, embedding_dim)

    # Test PositionalEncoding
    pos_enc = PositionalEncoding(embedding_dim)
    pos_embeddings = pos_enc(embeddings)
    print(f"Positional embeddings shape: {pos_embeddings.shape}")
    assert pos_embeddings.shape == (batch_size, lineup_size, embedding_dim)

    print("All player embedding modules working correctly!")


if __name__ == "__main__":
    main()
