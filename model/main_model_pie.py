import torch
from torch import nn
import numpy as np
from model.model_blocks import EmbedPosEnc, AttentionBlocks, Time_att
from model.FFN import FFN
from model.BottleNecks import Bottlenecks
from einops import repeat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # 可训练的损失缩放参数
        self.sigma_cls = nn.Parameter(torch.ones(1, 1, device=device))
        nn.init.kaiming_normal_(self.sigma_cls, mode='fan_out')
        self.sigma_reg = nn.Parameter(torch.ones(1, 1, device=device))
        nn.init.kaiming_normal_(self.sigma_reg, mode='fan_out')

        d_model = args.d_model
        hidden_dim = args.dff
        modal_nums = 2
        self.num_layers = args.num_layers

        # 从 args 读取 dropout 概率
        dropout_p = getattr(args, 'dropout', 0.1)
        # 两个 Dropout：一个放在 Attention 之后，一个放在 FFN 之后
        self.dropout_att = nn.Dropout(p=dropout_p)
        self.dropout_ffn = nn.Dropout(p=dropout_p)

        # 全局“绿色”token
        self.token = nn.Parameter(torch.ones(1, 1, d_model))

        # bbox 模态嵌入 + token
        self.bbox_embedding = EmbedPosEnc(args.bbox_input, d_model)
        self.bbox_token     = nn.Parameter(torch.ones(1, 1, d_model))

        # vel 模态嵌入 + token
        self.vel_embedding = EmbedPosEnc(args.vel_input, d_model)
        self.vel_token     = nn.Parameter(torch.ones(1, 1, d_model))

        # 各层的 Attention & FFN
        self.bbox_att  = nn.ModuleList()
        self.bbox_ffn  = nn.ModuleList()
        self.vel_att   = nn.ModuleList()
        self.vel_ffn   = nn.ModuleList()
        self.cross_att = nn.ModuleList()
        self.cross_ffn = nn.ModuleList()
        for _ in range(self.num_layers):
            self.bbox_att.append( AttentionBlocks(d_model, args.num_heads) )
            self.bbox_ffn.append( FFN(d_model, hidden_dim) )
            self.vel_att.append(  AttentionBlocks(d_model, args.num_heads) )
            self.vel_ffn.append(  FFN(d_model, hidden_dim) )
            self.cross_att.append( AttentionBlocks(d_model, args.num_heads) )
            self.cross_ffn.append( FFN(d_model, hidden_dim) )

        # 后续结构
        self.dense       = nn.Linear(modal_nums * d_model, 4)
        self.bottlenecks = Bottlenecks(d_model, args)
        self.time_att    = Time_att(dims=args.num_bnks)
        self.endp        = nn.Linear(modal_nums * d_model, 4)
        self.relu        = nn.ReLU()
        self.last        = nn.Linear(args.num_bnks, 1)
        self.sigmoid     = nn.Sigmoid()


    def forward(self, bbox, vel):
        """
        bbox: [B, T, 4]
        vel : [B, T, 2]
        """
        B = bbox.size(0)
        # 复制绿色 token
        token = repeat(self.token, '() s e -> b s e', b=B)

        # --- Embedding + 初始 Dropout
        bbox = self.bbox_embedding(bbox, self.bbox_token)
        bbox = self.dropout_att(bbox)
        vel  = self.vel_embedding(vel, self.vel_token)
        vel  = self.dropout_att(vel)

        # --- 第 1 层 Attention + FFN
        # bbox 自注意力
        bbox = self.bbox_att[0](bbox)
        bbox = self.dropout_att(bbox)
        token = torch.cat([token, bbox[:, 0:1, :]], dim=1)

        # vel 自注意力
        vel = self.vel_att[0](vel)
        vel = self.dropout_att(vel)
        token = torch.cat([token, vel[:, 0:1, :]], dim=1)

        # 交叉注意力
        token = self.cross_att[0](token)
        token = self.dropout_att(token)

        # 重构序列
        token_new = token[:, 0:1, :]
        bbox = torch.cat([token_new, bbox[:, 1:, :]], dim=1)
        vel  = torch.cat([token_new, vel[:, 1:, :]],  dim=1)

        # FFN + Dropout
        bbox = self.bbox_ffn[0](bbox)
        bbox = self.dropout_ffn(bbox)
        vel  = self.vel_ffn[0](vel)
        vel  = self.dropout_ffn(vel)

        token = self.cross_ffn[0](token)[:, 0:1, :]
        token = self.dropout_ffn(token)

        # --- 剩余层
        for i in range(1, self.num_layers):
            # bbox
            bbox = self.bbox_att[i](bbox)
            bbox = self.dropout_att(bbox)
            token = torch.cat([token, bbox[:, 0:1, :]], dim=1)

            # vel
            vel = self.vel_att[i](vel)
            vel = self.dropout_att(vel)
            token = torch.cat([token, vel[:, 0:1, :]], dim=1)

            # cross
            token = self.cross_att[i](token)
            token = self.dropout_att(token)

            # 重构
            token_new = token[:, 0:1, :]
            bbox = torch.cat([token_new, bbox[:, 1:, :]], dim=1)
            vel  = torch.cat([token_new, vel[:, 1:, :]],  dim=1)

            # FFN
            bbox = self.bbox_ffn[i](bbox)
            bbox = self.dropout_ffn(bbox)
            vel  = self.vel_ffn[i](vel)
            vel  = self.dropout_ffn(vel)

            token = self.cross_ffn[i](token)[:, 0:1, :]
            token = self.dropout_ffn(token)

        # --- 最终预测
        cls_out = torch.cat([bbox[:, 0:1, :], vel[:, 0:1, :]], dim=1)   # [B, 2, d_model]
        cls_flat = cls_out.flatten(start_dim=1)                         # [B, 2*d_model]
        end_point = self.endp(cls_flat)                                 # 端点回归

        bnk  = self.relu(self.time_att(self.bottlenecks(bbox, vel)))    # temporal bottleneck
        tmp  = self.last(bnk)                                           # 行人穿越意图
        pred = self.sigmoid(tmp)

        return pred, end_point, self.sigma_cls, self.sigma_reg