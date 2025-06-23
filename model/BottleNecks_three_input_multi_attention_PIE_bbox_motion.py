import torch
from torch import nn
import torch.nn.functional as F
# from model.model_blocks import EmbedPosEnc, AttentionBlocks, Time_att

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Bottlenecks(nn.Module):  # 瓶颈结构
    def __init__(self, dims, args):
        super(Bottlenecks, self).__init__()
        self.dims = dims
        self.num_bnks = args.num_bnks  # 单元数目
        self.num_layers = args.bnks_layers  # 层数
        # self.dropout_prob = getattr(args, 'dropout_prob', 0.5)

        # 分别为 bbox, vel, bbox_motion 建立线性层列表
        self.bbox = nn.ModuleList()
        self.vel = nn.ModuleList()
        self.bbox_motion = nn.ModuleList()  # <-- 新增的

        # 第 1 层
        self.bbox.append(nn.Linear(dims, dims + self.num_bnks, bias=True))
        self.vel.append(nn.Linear(dims, dims + self.num_bnks, bias=True))
        self.bbox_motion.append(nn.Linear(dims, dims + self.num_bnks, bias=True))  # <-- 新增的

        # 后续 (num_layers - 1) 层
        for _ in range(self.num_layers - 1):
            self.bbox.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
            self.vel.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
            self.bbox_motion.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))  # <-- 新增的

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # 新增：加性注意力层，用于融合 bnk_bbox, bnk_vel, bnk_bbox_motion
        self.attn = self.AdditiveAttention(input_dim=self.num_bnks)

    def cut(self, x):
        """
        将 x 从最后一维切片：
        x[:, :, :self.dims]   --> 主体部分
        x[:, :, -self.num_bnks:] --> bottlenecks 特征
        """
        return x[:, :, :self.dims], x[:, :, -self.num_bnks:]

    def forward(self, bbox, vel, bbox_motion):
        """
        同时处理 bbox, vel, bbox_motion 三个输入
        bbox, vel, bbox_motion 的形状一般是 [batch_size, seq_len, dims]
        """
        # =========== 第 1 层 =============
        # 1) bbox
        out_bbox = self.bbox[0](bbox)        # [B, T, dims] -> [B, T, dims+num_bnks]
        out_bbox = self.dropout(self.relu(out_bbox))
        bbox_main, bnk_bbox = self.cut(out_bbox)

        # 2) vel
        out_vel = self.vel[0](vel)
        out_vel = self.dropout(self.relu(out_vel))
        vel_main, bnk_vel = self.cut(out_vel)

        # 3) bbox_motion
        out_bbox_motion = self.bbox_motion[0](bbox_motion)
        out_bbox_motion = self.dropout(self.relu(out_bbox_motion))
        bbox_motion_main, bnk_bbox_motion = self.cut(out_bbox_motion)

        # # 将三者的 bottlenecks 部分进行合并
        # # 这里示例为简单加和：bottlenecks = bnk_bbox + bnk_vel + bnk_bbox_motion
        # # 你也可以改成 concat 或其他操作
        # bottlenecks = bnk_bbox + bnk_vel + bnk_bbox_motion


        # 用注意力融合三路 bottleneck
        bottlenecks = self.attn(bnk_bbox, bnk_vel, bnk_bbox_motion)  # [B, T, num_bnks]


        # =========== 后续层 (num_layers - 1) ===========
        for i in range(self.num_layers - 1):
            # bbox 的下一层输入：将主体部分 bbox_main 与 bottlenecks 拼接
            # 这里示例为在最后一维 cat
            cat_bbox = torch.cat((bbox_main, bottlenecks), dim=-1)
            out_bbox = self.bbox[i + 1](cat_bbox)
            out_bbox = self.dropout(self.relu(out_bbox))
            bbox_main, bnk_bbox = self.cut(out_bbox)

            # vel 的下一层输入
            cat_vel = torch.cat((vel_main, bottlenecks), dim=-1)
            out_vel = self.vel[i + 1](cat_vel)
            out_vel = self.dropout(self.relu(out_vel))
            vel_main, bnk_vel = self.cut(out_vel)

            # bbox_motion 的下一层输入
            cat_bbox_motion = torch.cat((bbox_motion_main, bottlenecks), dim=-1)
            out_bbox_motion = self.bbox_motion[i + 1](cat_bbox_motion)
            out_bbox_motion = self.dropout(self.relu(out_bbox_motion))
            bbox_motion_main, bnk_bbox_motion = self.cut(out_bbox_motion)

            # # 再将本层得到的三个 bottleneck 特征合并
            # bottlenecks = bnk_bbox + bnk_vel + bnk_bbox_motion
            # # 如果你想加上 bnk_token 之类的，也可以在这里进行
            # 再次用注意力融合
            bottlenecks = self.attn(bnk_bbox, bnk_vel, bnk_bbox_motion)

        # dims = bottlenecks.size(-1)
        # time_att = Time_att(dims)
        # bottlenecks = time_att(bottlenecks)  # 输出形状 [B, dims]

        return bottlenecks


    class AdditiveAttention(nn.Module):
        """
        PyTorch 实现的加性注意力机制，用于融合多个输入张量。
        """

        def __init__(self, input_dim):
            super().__init__()
            self.W_c = nn.Linear(input_dim, input_dim, bias=True)
            self.v = nn.Parameter(torch.randn(input_dim))

        def forward(self, *inputs):
            energies = []
            for x in inputs:
                # x: [B, T, input_dim]
                score = torch.tanh(self.W_c(x))  # [B, T, input_dim]
                e = torch.matmul(score, self.v)  # [B, T]
                energies.append(e)
            # 将多个得分堆叠为 [B, T, n_inputs]
            stacked = torch.stack(energies, dim=-1)
            alpha = F.softmax(stacked, dim=-1)  # [B, T, n_inputs]
            fused = 0
            for i, x in enumerate(inputs):
                fused += x * alpha[..., i].unsqueeze(-1)
            return fused

