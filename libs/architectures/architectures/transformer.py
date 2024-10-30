import torch
import torch.nn as nn
from torch import Tensor
from ml4gw.nn.norm import GroupNorm1DGetter, NormLayer
import numpy as np


class Pos_encoding(nn.Module):
    def __init__(self, d_model, maxlen, dropout):
        super(Pos_encoding, self).__init__()
        self.PE = torch.tensor([[pos / 1000**(i // 2 * 2 / d_model)
                               for pos in range(maxlen)] for i in range(d_model)])
        self.PE[:, 0::2] = np.sin(self.PE[:, 0::2])
        self.PE[:, 1::2] = np.cos(self.PE[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        out = self.dropout(input + self.PE)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        raw_input=True,
        in_channels=2,
        transformer_channels=64,
        num_series=3072,
        num_heads=8,
        encoder_layers=18
    ) -> None:
        super().__init__()
        self.raw_input = raw_input
        self.encoder_layers = encoder_layers
        self.num_heads=num_heads
        self.transformer_channels = transformer_channels

        print("\nHello! Now I am using the standard Transformer class, so you can see this info.\n")

        if raw_input:
            self.transformer_channels=2
            self.num_heads=2
            self.encoder_layers=4
        else: 
            self.conv1d = nn.Conv1d(
            in_channels=2,
            out_channels=self.transformer_channels,
            kernel_size=7,
            stride=3,
            padding=3,
            bias=False,
        )
            self.avgpool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)


        # data should be provided by format of (batch, seq, features)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_channels, nhead=self.num_heads, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.encoder_layers)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.transformer_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        print('')
        if not self.raw_input:
            x = self.conv1d(x)
            x = self.avgpool(x)
        
        # rearrange data dormat to (batch, seq, features)
        x = x.permute(0,2,1)
        x = self.transformer_encoder(x)

        # restore data dormat to (batch, features, seq)
        x = x.permute(0,2,1)
        # pooling and flatten
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
