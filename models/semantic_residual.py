import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticResidualScorerV2(nn.Module):
    def __init__(
        self,
        text_dim: int,
        num_relations: int,
        lambda_base: float = 0.05,
        text_norm: bool = True,
        delta_bound: float = 10.0,
        init_zero: bool = True,
    ):
        """
        v2: 关系自适应门控 (Relation-wise Gated)
        稳健增强:
        - text_norm: 对 h/r/t 文本向量做 L2 normalize，避免尺度漂移
        - delta_bound: 对 delta 做平滑有界化，避免极值把排序炸掉
            delta <- B * tanh(delta / B)
        - init_zero: 是否将 MLP 最后一层初始化为 0，使初始 delta≈0（更稳健）
        """
        super().__init__()
        self.text_dim = int(text_dim)
        self.lambda_base = float(lambda_base)

        self.text_norm = bool(text_norm)
        self.delta_bound = float(delta_bound)

        self.mlp = nn.Sequential(
            nn.Linear(self.text_dim * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # 每个关系一个 gate（标量）
        self.rel_gate = nn.Embedding(int(num_relations), 1)
        nn.init.zeros_(self.rel_gate.weight)

        # 关键稳健性：让初始 delta≈0，避免未训练的随机残差扰乱 baseline 排序
        if init_zero:
            last = self.mlp[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, h_text: torch.Tensor, r_text: torch.Tensor, t_text: torch.Tensor, rel_ids: torch.Tensor):
        """
        Args:
            h_text, r_text, t_text: [B, D]
            rel_ids: [B] (long)
        Returns:
            delta: [B]（已做 bounded）
            lambda_r: [B] 动态权重
        """
        # ---- 类型与形状兜底 ----
        if rel_ids.dtype != torch.long:
            rel_ids = rel_ids.long()
        rel_ids = rel_ids.view(-1)

        if self.text_norm:
            # eps 取 1e-8 更稳健（避免极小范数导致数值异常）
            h_text = F.normalize(h_text, p=2, dim=-1, eps=1e-8)
            r_text = F.normalize(r_text, p=2, dim=-1, eps=1e-8)
            t_text = F.normalize(t_text, p=2, dim=-1, eps=1e-8)

        feat = torch.cat(
            [
                h_text,
                r_text,
                t_text,
                h_text - t_text,
                h_text * t_text,
            ],
            dim=-1,
        )

        delta = self.mlp(feat).squeeze(-1)  # [B]

        # 平滑有界化：防止 delta 极端值把 s_struct + s_sem 搞崩
        if self.delta_bound > 0:
            B = self.delta_bound
            delta = B * torch.tanh(delta / B)

        # 关系门控
        w_r = self.rel_gate(rel_ids).squeeze(-1)  # [B]
        g_r = 2.0 * torch.sigmoid(w_r)            # (0, 2)
        lambda_r = self.lambda_base * g_r         # (0, 2*lambda_base)

        return delta, lambda_r
