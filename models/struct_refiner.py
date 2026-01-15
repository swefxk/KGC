import torch
import torch.nn as nn
import torch.nn.functional as F


class StructRefiner(nn.Module):
    def __init__(self, emb_dim: int, K: int, init_eta_raw: float = -6.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.K = K  # Max neighbors

        # 维度缩放参数 (a): 初始化为 1，保持原特征尺度
        self.a = nn.Parameter(torch.ones(emb_dim))

        # 残差强度 (eta): 初始化为极小值 (softplus(-6) ≈ 0.0025)
        # 保证训练初期 Refiner 几乎不干预，防止破坏 Baseline
        self.eta_raw = nn.Parameter(torch.tensor(init_eta_raw))

        # 频次门控参数: g = sigmoid(softplus(w)*log(freq) + b)
        # 频率越高的实体，g 越大 (倾向于关闭 Refiner)
        # 这里逻辑反转一下：我们希望 head 实体少 refine，tail 实体多 refine
        # 按照公式：e' = e + (1-g) * eta * m
        # 所以 g 越大 -> (1-g) 越小 -> refine 越弱。符合预期。
        self.w_raw = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def eta(self):
        return F.softplus(self.eta_raw)

    def refine_anchor(
            self,
            anchor_ids: torch.LongTensor,  # [B]
            rotate_model,  # 传入 RotatE 实例以获取 Embeddings
            nbr_ent: torch.LongTensor,  # [N,K]
            nbr_rel: torch.LongTensor,  # [N,K]
            nbr_dir: torch.BoolTensor,  # [N,K]
            nbr_mask: torch.BoolTensor,  # [N,K]
            freq: torch.Tensor  # [N]
    ) -> torch.Tensor:
        """
        返回 Refined 后的 Anchor Embedding [B, 2D]
        """
        device = anchor_ids.device
        d = self.emb_dim

        # 1. Gather 邻居信息
        # 支持 anchor_ids 包含重复索引 (Batch 内)
        a_ent = nbr_ent[anchor_ids]  # [B,K]
        a_rel = nbr_rel[anchor_ids]  # [B,K]
        a_dir = nbr_dir[anchor_ids]  # [B,K]
        a_msk = nbr_mask[anchor_ids]  # [B,K]

        # 2. 获取邻居实体的 Embedding
        e_j = rotate_model.entity_embedding[a_ent]  # [B,K,2d]
        re_j, im_j = torch.chunk(e_j, 2, dim=-1)  # [B,K,d]

        # 3. 获取关系的 Embedding (Real/Imag)
        # 注意：这里调用的是 rotate_model.get_relation_reim，它现在支持处理 [B,K] 维度的输入
        re_r, im_r = rotate_model.get_relation_reim(a_rel, conj=False)  # [B,K,d]

        # 构造共轭关系 (用于 OUT 边)
        # 原理：如果 i -> j (OUT)，则 j 对 i 的贡献应该是 j * conj(r)
        # 如果 j -> i (IN)，则 j 对 i 的贡献应该是 j * r
        re_rc, im_rc = re_r, -im_r

        # 根据方向选择 r 或 conj(r)
        # a_dir: True=OUT, False=IN
        # unsqueeze(-1) -> [B,K,1]
        re_sel = torch.where(a_dir.unsqueeze(-1), re_rc, re_r)
        im_sel = torch.where(a_dir.unsqueeze(-1), im_rc, im_r)

        # 4. 消息传递 (Message Passing): msg = e_j * r_sel
        re_msg = re_j * re_sel - im_j * im_sel
        im_msg = re_j * im_sel + im_j * re_sel

        # 5. Mask 掉无效邻居 (Padding)
        msk = a_msk.unsqueeze(-1)  # [B,K,1]
        re_msg = re_msg * msk
        im_msg = im_msg * msk

        # 6. 聚合 (Mean Aggregation)
        deg = a_msk.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
        re_m = re_msg.sum(dim=1) / deg  # [B,d]
        im_m = im_msg.sum(dim=1) / deg

        # 7. 频次门控 (Frequency Gating)
        f = freq[anchor_ids].to(device)  # [B]
        x = torch.log1p(f)  # log(freq + 1)
        w = F.softplus(self.w_raw)
        g = torch.sigmoid(w * x + self.b).unsqueeze(-1)  # [B,1]

        # 8. 计算残差并叠加
        # e' = e + (1-g) * eta * (a * m)
        eta = self.eta()
        a = self.a.unsqueeze(0)  # [1,d]

        re_delta = (1.0 - g) * eta * (a * re_m)
        im_delta = (1.0 - g) * eta * (a * im_m)

        # 获取原始 Anchor Embedding
        e_i = rotate_model.entity_embedding[anchor_ids]
        re_i, im_i = torch.chunk(e_i, 2, dim=-1)

        re_out = re_i + re_delta
        im_out = im_i + im_delta

        return torch.cat([re_out, im_out], dim=-1)