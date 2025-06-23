import torch
from torch import nn
import numpy as np
from model.model_blocks import EmbedPosEnc, AttentionBlocks, Time_att, TimeTransformer
from model.FFN import FFN
from model.BottleNecks_three_input_multi_attention import Bottlenecks
from einops import repeat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.sigma_cls = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) # 生成一个可训练的分类损失参数
        nn.init.kaiming_normal_(self.sigma_cls, mode='fan_out') # 初始化参数
        self.sigma_reg = nn.Parameter(torch.ones(1, 1, requires_grad=True, device=device), requires_grad=True) # 生成一个可训练的回归损失参数
        nn.init.kaiming_normal_(self.sigma_reg, mode='fan_out')  # 初始化参数

        # 1) 定义 Dropout，p=0.5 只是示例，你可调成 0.2、0.3 等
        self.dropout_att = nn.Dropout(p=0.5)  # 用于 Attention 输出后
        self.dropout_ffn = nn.Dropout(p=0.5)  # 用于 FFN 输出后

        d_model = args.d_model
        hidden_dim = args.dff
        modal_nums = 3
        self.num_layers = args.num_layers
        # self.token = nn.Parameter(torch.ones(1, 1, d_model)) # 生成一个可训练的token @@绿色token

        self.bbox_embedding = EmbedPosEnc(args.bbox_input, d_model) # 张量嵌入以及生成位置编码
        self.bbox_token = nn.Parameter(torch.ones(1, 1, d_model))   # 生成一个可训练的bbox_token

        self.vel_embedding = EmbedPosEnc(args.vel_input, d_model)  # 张量嵌入以及生成位置编码
        self.vel_token = nn.Parameter(torch.ones(1, 1, d_model))   # 生成一个可训练的vel_token

        # ---ADDED: 加速度的嵌入与 token
        # 需要在 args 中添加 args.acc_input（例如 3 表示 x/y/z 三维加速度）
        self.acc_embedding = EmbedPosEnc(args.acc_input, d_model)
        self.acc_token = nn.Parameter(torch.ones(1, 1, d_model))

        self.bbox_att = nn.ModuleList() # 生成一个空的ModuleList
        self.bbox_ffn = nn.ModuleList()
        self.vel_att = nn.ModuleList()
        self.vel_ffn = nn.ModuleList()
        # self.cross_att = nn.ModuleList()
        # self.cross_ffn = nn.ModuleList()

        # ---ADDED: 为加速度添加对应的注意力与 FFN
        self.acc_att = nn.ModuleList()
        self.acc_ffn = nn.ModuleList()

        for _ in range(self.num_layers):
            self.bbox_att.append(AttentionBlocks(d_model, args.num_heads)) # 添加AttentionBlocks
            self.bbox_ffn.append(FFN(d_model, hidden_dim)) # 添加FFN
            self.vel_att.append(AttentionBlocks(d_model, args.num_heads))
            self.vel_ffn.append(FFN(d_model, hidden_dim))

            # acc
            self.acc_att.append(AttentionBlocks(d_model, args.num_heads))  # ---ADDED
            self.acc_ffn.append(FFN(d_model, hidden_dim))  # ---ADDED

            # self.cross_att.append(AttentionBlocks(d_model, args.num_heads)) # 添加AttentionBlocks
            # self.cross_ffn.append(FFN(d_model, hidden_dim))

        self.dense = nn.Linear(modal_nums * d_model, 4) # 全连接层
        self.bottlenecks = Bottlenecks(d_model, args) # Bottlenecks
        self.time_att = Time_att(dims=args.num_bnks) # Time_att
        self.time_transformer = TimeTransformer(d_model=args.num_bnks, num_heads=args.time_transformer_num_heads, dropout=args.time_transformer_dropout)
        self.endp = nn.Linear(modal_nums * d_model, 4) # 全连接层
        self.relu = nn.ReLU()
        self.last = nn.Linear(args.num_bnks, 1) # 全连接层
        self.sigmoid = nn.Sigmoid() # sigmoid激活函数

    def forward(self, bbox, vel, acc):
        '''
            :bbox       :[b, 4, 32]
            :vel        :[b, 2, 32]            ????????????????????????????
            :param acc:  [b, 3, 32]  (batch_size, acc_channels, seq_len)
        '''
        '''
            bbox: [64, 16, 4]
            vel: [64, 16, 2]
            acc: [64, 16, 3]
        '''
        b = bbox.shape[0]
        # token = repeat(self.token, '() s e -> b s e', b=b) # 重复token，使尺寸匹配

        bbox = self.bbox_embedding(bbox, self.bbox_token) # 张量嵌入以及生成位置编码
        vel = self.vel_embedding(vel, self.vel_token) # 张量嵌入以及生成位置编码
        acc = self.acc_embedding(acc, self.acc_token)

        bbox = self.bbox_att[0](bbox) # bbox的自注意力

        # 在这里对 bbox 的输出做 dropout
        bbox = self.dropout_att(bbox)

        # token = torch.cat([token, bbox[:, 0:1, :]], dim=1)  # 拼接token和bbox
        vel = self.vel_att[0](vel) # vel的自注意力
        #dropout
        vel = self.dropout_att(vel)

        # token = torch.cat([token, vel[:, 0:1, :]], dim=1) # 拼接token和vel

        # 3) acc 自注意力  # ---ADDED
        acc = self.acc_att[0](acc)
        acc = self.dropout_att(acc) #dropout

        # token = torch.cat([token, acc[:, 0:1, :]], dim=1)

        # token = self.cross_att[0](token) # token的交叉注意力
        # token = self.dropout_att(token) #dropout

        # # token_new = token[:, 0:1, :] # 取出token的第一个元素
        # bbox = torch.cat([token_new, bbox[:, 1:, :]], dim=1) # 拼接token_new和bbox
        # vel = torch.cat([token_new, vel[:, 1:, :]], dim=1) # 拼接token_new和vel
        # acc = torch.cat([token_new, acc[:, 1:, :]], dim=1)  # ---ADDED

        # === 第 1 层 FFN === dropout
        bbox = self.bbox_ffn[0](bbox)
        bbox = self.dropout_ffn(bbox)

        vel = self.bbox_ffn[0](vel)
        vel = self.dropout_ffn(vel)

        acc = self.acc_ffn[0](acc)
        acc = self.dropout_ffn(acc)

        # token = self.cross_ffn[0](token)[:, 0:1, :]
        # token = self.dropout_ffn(token)


        # bbox = self.bbox_ffn[0](bbox) # bbox的FFN
        # vel = self.vel_ffn[0](vel) # vel的FFN
        #
        # acc = self.acc_ffn[0](acc)  # ---ADDED
        #
        # token = self.cross_ffn[0](token)[:, 0:1, :] # token的FFN

        for i in range(self.num_layers - 1):
            bbox = self.bbox_att[i + 1](bbox)
            bbox = self.dropout_att(bbox)
            # token = torch.cat([token, bbox[:, 0:1, :]], dim=1)
            vel = self.vel_att[i + 1](vel)
            vel = self.dropout_att(vel) #dropout
            # token = torch.cat([token, vel[:, 0:1, :]], dim=1)

            # 3) acc  # ---ADDED
            acc = self.acc_att[i + 1](acc)
            acc = self.dropout_att(acc) #dropout
            # token = torch.cat([token, acc[:, 0:1, :]], dim=1)

            # token = self.cross_att[i + 1](token)
            # token = self.dropout_att(token) #dropout

            # token_new = token[:, 0:1, :]
            # bbox = torch.cat([token_new, bbox[:, 1:, :]], dim=1)
            # vel = torch.cat([token_new, vel[:, 1:, :]], dim=1)
            #
            # acc = torch.cat([token_new, acc[:, 1:, :]], dim=1)  # ---ADDED

            bbox = self.bbox_ffn[i + 1](bbox)
            bbox = self.dropout_ffn(bbox)
            vel = self.vel_ffn[i + 1](vel)
            vel = self.dropout_ffn(vel)

            acc = self.acc_ffn[i + 1](acc)  # ---ADDED
            acc = self.dropout_ffn(acc)

            # token = self.cross_ffn[i + 1](token)[:, 0:1, :]
            # token = self.dropout_ffn(token)


        cls_out = torch.cat([bbox[:, 0:1, :], vel[:, 0:1, :], acc[:, 0:1, :]], dim=1) # 拼接bbox的token和vel的token
        cls_out_flatten = torch.flatten(cls_out, start_dim=1) # 展平
        end_point = self.endp(cls_out_flatten) # 全连接层预测endpoint

        # bnk = self.relu(self.time_att(self.bottlenecks(bbox, vel, acc))) # Bottlenecks
        bnk = self.relu(self.time_transformer(self.bottlenecks(bbox, vel, acc)))  # Bottlenecks
        tmp = self.last(bnk) # 全连接层预测穿越行为
        pred = self.sigmoid(tmp)
        return pred, end_point, self.sigma_cls, self.sigma_reg # 返回预测结果，endpoint预测结果，分类的sigma，回归的sigma
