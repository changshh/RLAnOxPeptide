#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class AntioxidantPredictor(nn.Module):
    def __init__(self, input_dim, transformer_layers=3, transformer_heads=4, transformer_dropout=0.1):
        super(AntioxidantPredictor, self).__init__()
        self.prott5_dim = 1024
        self.handcrafted_dim = input_dim - self.prott5_dim
        self.seq_len = 16
        self.prott5_feature_dim = 64

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.prott5_feature_dim,
            nhead=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        fused_dim = self.prott5_feature_dim + self.handcrafted_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x, *args):
        batch_size = x.size(0)
        prot_t5_features = x[:, : self.prott5_dim]
        handcrafted_features = x[:, self.prott5_dim :]

        prot_t5_seq = prot_t5_features.view(batch_size, self.seq_len, self.prott5_feature_dim)
        encoded_seq = self.transformer_encoder(prot_t5_seq)
        refined_prott5 = encoded_seq.mean(dim=1)

        fused_features = torch.cat([refined_prott5, handcrafted_features], dim=1)
        fused_features = self.fusion_fc(fused_features)
        logits = self.classifier(fused_features)
        return logits / self.temperature

    def set_temperature(self, temp_value, device):
        """Set the calibrated temperature value."""
        self.temperature = nn.Parameter(torch.tensor([temp_value], device=device), requires_grad=False)
        print(f"Model temperature set to: {self.temperature.item()}")

    def get_temperature(self):
        """Return the current temperature value."""
        return self.temperature.item()
