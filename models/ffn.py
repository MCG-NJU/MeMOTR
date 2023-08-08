# Copyright (c) Ruopeng Gao. All Rights Reserved.
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout: float):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt):
        tgt2 = self.linear2(
            self.dropout1(
                self.activation(
                    self.linear1(tgt)
                )
            )
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt
