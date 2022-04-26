from torch import nn
from einops.layers.torch import Rearrange

from enformer_pytorch import Enformer, Residual
from performer_pytorch import SelfAttention

class EnformerPerformer(Enformer):
    def __init__(self, config):
        super(Enformer, self, config).__init__()
        # transformer
        self.transformer = []
        for _ in range(config.depth):
            self.transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    SelfAttention(
                        dim=config.dim,
                        heads=config.heads,
                        dropout=config.attn_dropout
                    ),
                    # difference from Attention module -> no relative positional embedding options, no seperate dimensions for key and value
                    nn.Dropout(config.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                ))
            ))

        self.transformer = nn.Sequential(
            Rearrange('b d n -> b n d'),
            *self.transformer
        )