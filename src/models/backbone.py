"""
Transformer backbone for PGC squad modeling.
"""

import torch
import torch.nn as nn

from src.models.modules import (
    FourierPositionalEncoding,
    ZoneEmbedding,
    TokenEmbedding,
)
from src.data.dataset import NUM_MAPS


class TransformerBackbone(nn.Module):
    """
    Transformer backbone for squad-level modeling.

    Takes continuous features from all squads, embeds them into tokens,
    and applies self-attention to capture inter-squad interactions.
    Also incorporates position information and zone embeddings.

    Args:
        input_dim: Number of input continuous features.
        embed_dim: Dimension of token embeddings.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        mlp_hidden_dim: Hidden dimension in token embedding MLP.
        ff_hidden_dim: Hidden dimension in transformer feedforward layers.
        dropout: Dropout probability.
        num_frequencies: Number of frequency bands for positional encoding.
        encoder_type: 'transformer' (self-attention over squads) or 'mlp'
            (per-squad MLP with no cross-squad attention; used for the
            encoder ablation in Table 2). Same features, losses, and
            training settings apply to both.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_hidden_dim: int = None,
        ff_hidden_dim: int = None,
        dropout: float = 0.1,
        num_frequencies: int = 8,
        encoder_type: str = "transformer",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.encoder_type = encoder_type.lower()
        assert self.encoder_type in ("transformer", "mlp"), \
            f"encoder_type must be 'transformer' or 'mlp', got {encoder_type}"

        # Token embedding layer (continuous features)
        self.token_embedding = TokenEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout,
        )

        # 3D positional encoding for player positions
        self.position_encoding = FourierPositionalEncoding(
            coord_dim=3,
            embed_dim=embed_dim,
            num_frequencies=num_frequencies,
        )

        # Zone embeddings
        self.bluezone_embedding = ZoneEmbedding(
            embed_dim=embed_dim,
            num_frequencies=num_frequencies,
        )
        self.whitezone_embedding = ZoneEmbedding(
            embed_dim=embed_dim,
            num_frequencies=num_frequencies,
        )

        # Map embedding (4 maps: erangel, miramar, taego, rondo)
        self.map_embedding = nn.Embedding(NUM_MAPS, embed_dim)

        # Encoder (Transformer vs per-squad MLP)
        if ff_hidden_dim is None:
            ff_hidden_dim = embed_dim * 4

        if self.encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_hidden_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )
        else:
            # MLP encoder: each squad token is processed independently,
            # so no inter-squad interaction is modeled.
            layers = []
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(embed_dim, ff_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_hidden_dim, embed_dim),
                    nn.Dropout(dropout),
                ])
            self.transformer = nn.Sequential(*layers)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def _pool_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Average pool alive player positions per squad.

        Args:
            positions: (batch, num_squads, 4, 3) - player positions
                       nan values indicate dead players

        Returns:
            (batch, num_squads, embed_dim) - position embeddings
        """
        batch, num_squads, num_players, coord_dim = positions.shape

        # Create mask for alive players (not nan)
        # (batch, num_squads, 4)
        alive_mask = ~torch.isnan(positions[..., 0])

        # Replace nan with 0 for computation
        positions_clean = positions.clone()
        positions_clean[torch.isnan(positions_clean)] = 0.0

        # Encode all positions: (batch, num_squads, 4, embed_dim)
        pos_emb = self.position_encoding(positions_clean)

        # Mask out dead players: (batch, num_squads, 4, 1)
        mask_expanded = alive_mask.unsqueeze(-1).float()

        # Weighted sum and count
        # (batch, num_squads, embed_dim)
        pos_sum = (pos_emb * mask_expanded).sum(dim=2)
        count = mask_expanded.sum(dim=2).clamp(min=1)  # Avoid division by zero

        # Average pool
        pos_pooled = pos_sum / count

        return pos_pooled

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        bluezone_info: torch.Tensor,
        whitezone_info: torch.Tensor,
        map_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, num_squads, input_dim) - continuous features
            positions: (batch, num_squads, 4, 3) - player positions
            bluezone_info: (batch, 3) - [x, y, radius]
            whitezone_info: (batch, 3) - [x, y, radius]
            map_idx: (batch,) - map index (0-3)

        Returns:
            (batch, num_squads, embed_dim) - squad embeddings
        """
        # Token embedding: (batch, num_squads, embed_dim)
        tokens = self.token_embedding(x)

        # Add position embeddings
        pos_emb = self._pool_positions(positions)
        tokens = tokens + pos_emb

        # Add zone embeddings (broadcast to all squads)
        blue_emb = self.bluezone_embedding(bluezone_info)
        tokens = tokens + blue_emb.unsqueeze(1)

        white_emb = self.whitezone_embedding(whitezone_info)
        tokens = tokens + white_emb.unsqueeze(1)

        # Add map embedding (broadcast to all squads)
        # map_idx: (batch,) -> map_emb: (batch, embed_dim)
        map_emb = self.map_embedding(map_idx)
        tokens = tokens + map_emb.unsqueeze(1)

        # Transformer encoding
        encoded = self.transformer(tokens)

        # Layer normalization
        output = self.norm(encoded)

        return output


if __name__ == "__main__":
    # Test
    batch_size = 4
    num_squads = 16
    input_dim = 53

    model = TransformerBackbone(
        input_dim=input_dim,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
    )

    # Create dummy inputs
    x = torch.randn(batch_size, num_squads, input_dim)
    positions = torch.randn(batch_size, num_squads, 4, 3)
    positions[0, 0, 1, :] = float('nan')  # Simulate dead player
    bluezone = torch.randn(batch_size, 3)
    whitezone = torch.randn(batch_size, 3)
    map_idx = torch.randint(0, NUM_MAPS, (batch_size,))

    # Forward pass
    output = model(x, positions, bluezone, whitezone, map_idx)

    print(f"Input shape: {x.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Bluezone shape: {bluezone.shape}")
    print(f"Whitezone shape: {whitezone.shape}")
    print(f"Map index shape: {map_idx.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

