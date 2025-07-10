#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class AntioxidantPredictor(nn.Module):
    def __init__(self, input_dim, transformer_layers=3, transformer_heads=4, transformer_dropout=0.1):
        super(AntioxidantPredictor, self).__init__()
        self.prott5_dim = 1024
        self.handcrafted_dim = input_dim - self.prott5_dim
        self.seq_len = 16
        self.prott5_feature_dim = 64  # 16 * 64 = 1024

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.prott5_feature_dim,
            nhead=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        fused_dim = self.prott5_feature_dim + self.handcrafted_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # 温度缩放参数 T
        # 初始化为1.0，表示在校准前不改变logits
        # requires_grad=False，因为T通常在模型训练完成后单独优化
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x, *args):
        batch_size = x.size(0)
        prot_t5_features = x[:, :self.prott5_dim]
        handcrafted_features = x[:, self.prott5_dim:]
        
        prot_t5_seq = prot_t5_features.view(batch_size, self.seq_len, self.prott5_feature_dim)
        encoded_seq = self.transformer_encoder(prot_t5_seq)
        refined_prott5 = encoded_seq.mean(dim=1)
        
        fused_features = torch.cat([refined_prott5, handcrafted_features], dim=1)
        fused_features = self.fusion_fc(fused_features)
        
        logits = self.classifier(fused_features)
        
        # 应用温度缩放: logits / T
        # 注意：这里是在获取原始logits后，外部应用sigmoid前进行缩放
        # 如果要直接输出校准后的概率，可以在这里除以T然后sigmoid
        # 但通常T的优化和应用是分离的。
        # 为了在调用模型时就能获得校准的logits（如果T已优化），我们在这里应用它。
        # 如果T未被优化（仍为1），则此操作无影响。
        logits_scaled = logits / self.temperature
        
        return logits_scaled # 返回校准后（或原始，如果T=1）的logits

    def set_temperature(self, temp_value, device):
        """用于设置优化后的温度值"""
        self.temperature = nn.Parameter(torch.tensor([temp_value], device=device), requires_grad=False)
        print(f"模型温度 T 设置为: {self.temperature.item()}")

    def get_temperature(self):
        """获取当前温度值"""
        return self.temperature.item()

if __name__ == "__main__":
    dummy_input = torch.randn(8, 1914)
    model = AntioxidantPredictor(input_dim=1914)
    
    print(f"初始温度: {model.get_temperature()}")
    logits_output_initial = model(dummy_input)
    print("初始 logits shape:", logits_output_initial.shape)
    probs_initial = torch.sigmoid(logits_output_initial)
    print("初始概率 (T=1.0):", probs_initial.detach().cpu().numpy()[:2])

    # 模拟设置一个优化后的温度
    model.set_temperature(1.5, device='cpu') # 假设优化得到 T=1.5
    print(f"设置后温度: {model.get_temperature()}")
    logits_output_scaled = model(dummy_input) # 模型内部应用了 T
    print("缩放后 logits shape:", logits_output_scaled.shape)
    probs_scaled = torch.sigmoid(logits_output_scaled) # 外部仍然需要 sigmoid
    print("缩放后概率 (T=1.5):", probs_scaled.detach().cpu().numpy()[:2])
    
    # 验证 logits / T 的效果
    # logits_manual_scale = logits_output_initial / 1.5
    # probs_manual_scale = torch.sigmoid(logits_manual_scale)
    # print("手动缩放后概率 (T=1.5):", probs_manual_scale.detach().cpu().numpy()[:2])
    # assert torch.allclose(probs_scaled, probs_manual_scale) # 应该相等
