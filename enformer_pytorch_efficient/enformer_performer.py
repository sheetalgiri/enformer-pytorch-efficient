from torch import nn
from einops.layers.torch import Rearrange

from enformer_pytorch import Enformer, Residual
from enformer_pytorch.config_enformer import EnformerConfig
from performer_pytorch import SelfAttention

class EnformerPerformer(Enformer):
    def from_hparams(**kwargs):
        return EnformerPerformer(EnformerConfig(**kwargs))

    def __init__(self, config):
        super().__init__(config)
        # transformer
        transformer = []
        for _ in range(config.depth):
            transformer.append(nn.Sequential(
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
            *transformer
        )

        self._trunk = nn.Sequential(
            self.stem,
            self.conv_tower,
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )