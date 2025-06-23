# model/main_model_xyz_acc_dropout_multi_head_attention_QKV_BFF.py

import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bbox_embed = nn.Linear(4, args.d_model)
        self.pos_enc    = PositionalEncoding(args.d_model)
        self.encoder    = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.num_heads,
                dim_feedforward=args.dff,
                dropout=args.time_transformer_dropout
            ),
            num_layers=args.num_layers
        )
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(args.d_model, 1)

    def forward(self, bbox, lengths):
        """
        bbox:    [B, T_max, 4]  —— 归一化后 + 0 填充
        lengths: [B]            —— 每条序列真实长度
        """
        x = self.bbox_embed(bbox)             # [B,T,d_model]
        x = self.pos_enc(x)                   # 加位置信息
        # padding mask: True 表示 pad 的位置
        max_len = bbox.size(1)
        arange  = torch.arange(max_len, device=bbox.device).unsqueeze(0)
        padding_mask = arange >= lengths.unsqueeze(1)  # [B, T_max]
        x = x.transpose(0,1)                 # -> [T,B,E]
        out = self.encoder(x, src_key_padding_mask=padding_mask)
        out = out.transpose(0,1)             # -> [B,T,E]
        out = self.pool(out.transpose(1,2)).squeeze(-1)  # [B,E]
        logits = self.classifier(out)         # [B,1]
        return torch.sigmoid(logits).squeeze(-1)  # [B]
