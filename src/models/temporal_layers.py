"""
Temporal Layers for Possession Sequence Modeling

Implements:
1. LSTM layers for long-range dependency capture
2. Temporal Convolutional Networks (TCN) for multi-scale patterns
3. Hybrid temporal architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class TemporalConvBlock(nn.Module):
    """
    Single temporal convolutional block with residual connections.

    Features:
    - Dilated causal convolutions
    - Residual connections
    - Weight normalization
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.2):
        """
        Initialize temporal conv block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor for receptive field
            dropout: Dropout probability
        """
        super(TemporalConvBlock, self).__init__()

        # Ensure causal convolutions (only look at past)
        self.padding = (kernel_size - 1) * dilation

        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.conv1 = nn.utils.weight_norm(self.conv1)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.conv2 = nn.utils.weight_norm(self.conv2)

        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, channels, seq_len)

        Returns:
            Output tensor, shape (batch, channels, seq_len)
        """
        # First conv block
        out = self.conv1(x)
        # Remove future-looking padding
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNetwork(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling.

    Stacks multiple temporal conv blocks with increasing dilation
    to capture multi-scale temporal patterns.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_channels: int = 128,
                 num_layers: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.2):
        """
        Initialize TCN.

        Args:
            input_dim: Input feature dimension
            hidden_channels: Number of channels in hidden layers
            num_layers: Number of TCN layers
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super(TemporalConvNetwork, self).__init__()

        layers = []
        num_channels = [input_dim] + [hidden_channels] * num_layers

        for i in range(num_layers):
            dilation = 2 ** i  # Exponentially increasing dilation
            layers.append(
                TemporalConvBlock(
                    in_channels=num_channels[i],
                    out_channels=num_channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Output tensor, shape (batch, seq_len, hidden_channels)
        """
        # TCN expects (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Transpose back to (batch, seq_len, features)
        out = out.transpose(1, 2)
        return out


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for possession sequence encoding.

    Note: In practice for real-time prediction, use unidirectional LSTM.
    Bidirectional is used here for offline analysis and maximum accuracy.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True):
        """
        Initialize LSTM.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(BidirectionalLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output dimension
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self,
                x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, features)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (output, (h_n, c_n))
            - output: shape (batch, seq_len, hidden_dim * num_directions)
            - h_n: shape (num_layers * num_directions, batch, hidden_dim)
            - c_n: shape (num_layers * num_directions, batch, hidden_dim)
        """
        output, hidden = self.lstm(x, hidden)
        return output, hidden


class HybridTemporalEncoder(nn.Module):
    """
    Hybrid temporal encoder combining LSTM and TCN.

    Architecture:
    1. TCN branch: Captures multi-scale patterns
    2. LSTM branch: Captures long-range dependencies
    3. Fusion layer: Combines both representations
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_lstm_layers: int = 2,
                 num_tcn_layers: int = 4,
                 dropout: float = 0.2,
                 fusion_method: str = 'concat'):
        """
        Initialize hybrid encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for both LSTM and TCN
            num_lstm_layers: Number of LSTM layers
            num_tcn_layers: Number of TCN layers
            dropout: Dropout probability
            fusion_method: How to fuse LSTM and TCN ('concat', 'add', 'gated')
        """
        super(HybridTemporalEncoder, self).__init__()

        self.fusion_method = fusion_method

        # LSTM branch
        self.lstm = BidirectionalLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,  # Divide by 2 because bidirectional
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=True
        )

        # TCN branch
        self.tcn = TemporalConvNetwork(
            input_dim=input_dim,
            hidden_channels=hidden_dim,
            num_layers=num_tcn_layers,
            dropout=dropout
        )

        # Fusion layer
        if fusion_method == 'concat':
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_method == 'add':
            self.fusion = None  # Simple addition
        elif fusion_method == 'gated':
            # Gated fusion mechanism
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, features)

        Returns:
            Encoded representation, shape (batch, seq_len, hidden_dim)
        """
        # LSTM branch
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # TCN branch
        tcn_out = self.tcn(x)  # (batch, seq_len, hidden_dim)

        # Fusion
        if self.fusion_method == 'concat':
            combined = torch.cat([lstm_out, tcn_out], dim=-1)  # (batch, seq_len, 2*hidden_dim)
            fused = self.fusion(combined)  # (batch, seq_len, hidden_dim)
        elif self.fusion_method == 'add':
            fused = lstm_out + tcn_out
        elif self.fusion_method == 'gated':
            combined = torch.cat([lstm_out, tcn_out], dim=-1)
            gate = self.gate(combined)
            fused = gate * lstm_out + (1 - gate) * tcn_out

        # Normalize and dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


def main():
    """Test temporal layers."""
    batch_size = 32
    seq_len = 50
    input_dim = 64
    hidden_dim = 128

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)

    # Test TCN
    print("Testing TCN...")
    tcn = TemporalConvNetwork(input_dim, hidden_dim, num_layers=4)
    tcn_out = tcn(x)
    print(f"TCN output shape: {tcn_out.shape}")
    assert tcn_out.shape == (batch_size, seq_len, hidden_dim)

    # Test LSTM
    print("Testing LSTM...")
    lstm = BidirectionalLSTM(input_dim, hidden_dim // 2, num_layers=2)
    lstm_out, _ = lstm(x)
    print(f"LSTM output shape: {lstm_out.shape}")
    assert lstm_out.shape == (batch_size, seq_len, hidden_dim)

    # Test Hybrid Encoder
    print("Testing Hybrid Encoder...")
    hybrid = HybridTemporalEncoder(input_dim, hidden_dim, fusion_method='concat')
    hybrid_out = hybrid(x)
    print(f"Hybrid output shape: {hybrid_out.shape}")
    assert hybrid_out.shape == (batch_size, seq_len, hidden_dim)

    print("All temporal layers working correctly!")


if __name__ == "__main__":
    main()
