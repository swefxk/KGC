import torch
import torch.nn as nn
import torch.nn.functional as F


class StructRefiner(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        K: int,
        num_relations: int,
        init_eta_raw: float = -6.0,
        attn_dim: int = 128,
        eta_max: float = 0.5,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.K = K  # Max neighbors
        self.num_relations = num_relations
        self.attn_dim = attn_dim
        self.eta_max = float(eta_max)

        # 维度缩放参数 (a): 初始化为 1，保持原特征尺度
        self.a = nn.Parameter(torch.ones(emb_dim))

        # 残差强度 (eta): 初始化为极小值 (softplus(-6) ≈ 0.0025)
        # 保证训练初期 Refiner 几乎不干预，防止破坏 Baseline
        self.eta_raw = nn.Parameter(torch.tensor(init_eta_raw))

        # 频次门控参数：eta = eta_max * sigmoid(eta_raw + softplus(w_raw) * (-logfreq) + b)
        # 高频(Head) -> eta 更小；低频(Tail) -> eta 更大
        self.w_raw = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

        # Directional relation-aware attention (query/key projection)
        in_dim = 2 * emb_dim
        self.w_q = nn.Linear(in_dim, attn_dim, bias=False)
        self.w_k = nn.Linear(in_dim, attn_dim, bias=False)
        nn.init.zeros_(self.w_q.weight)
        nn.init.zeros_(self.w_k.weight)

        # Relation / direction bias
        self.rel_bias = nn.Embedding(num_relations, 1)
        self.dir_bias = nn.Embedding(2, 1)  # OUT=1, IN=0
        nn.init.zeros_(self.rel_bias.weight)
        nn.init.zeros_(self.dir_bias.weight)

        # Candidate-aware delta head (zero-init for cold start)
        self.q_proj = nn.Linear(in_dim, attn_dim, bias=False)
        self.t_proj = nn.Linear(in_dim, attn_dim, bias=False)
        # cold start: keep delta near 0 but allow gradients to flow
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.t_proj.weight)
        self.rel_bias_vec = nn.Embedding(num_relations, attn_dim)
        self.dir_bias_vec = nn.Embedding(2, attn_dim)
        nn.init.zeros_(self.rel_bias_vec.weight)
        nn.init.zeros_(self.dir_bias_vec.weight)

    def eta(self):
        return F.softplus(self.eta_raw)

    def _compute_ctx(
            self,
            anchor_ids: torch.LongTensor,  # [B]
            rotate_model,
            nbr_ent: torch.LongTensor,  # [N,K]
            nbr_rel: torch.LongTensor,  # [N,K]
            nbr_dir: torch.BoolTensor,  # [N,K]
            nbr_mask: torch.BoolTensor,  # [N,K]
    ):
        """
        Return neighbor-aggregated context (delta) and has_nbr mask.
        """
        device = anchor_ids.device
        d = self.emb_dim

        # 1. Gather neighbor info
        a_ent = nbr_ent[anchor_ids]  # [B,K]
        a_rel = nbr_rel[anchor_ids]  # [B,K]
        a_dir = nbr_dir[anchor_ids]  # [B,K]
        a_msk = nbr_mask[anchor_ids]  # [B,K]

        # 2. Neighbor entity embeddings
        e_j = rotate_model.entity_embedding[a_ent]  # [B,K,2d]
        re_j, im_j = torch.chunk(e_j, 2, dim=-1)  # [B,K,d]

        # 3. Relation embeddings
        re_r, im_r = rotate_model.get_relation_reim(a_rel, conj=False)  # [B,K,d]
        re_rc, im_rc = re_r, -im_r

        # 4. Directional selection
        re_sel = torch.where(a_dir.unsqueeze(-1), re_rc, re_r)
        im_sel = torch.where(a_dir.unsqueeze(-1), im_rc, im_r)

        # 5. Message passing (complex multiply)
        re_msg = re_j * re_sel - im_j * im_sel
        im_msg = re_j * im_sel + im_j * re_sel
        hat = torch.cat([re_msg, im_msg], dim=-1)  # [B,K,2d]

        # 6. Directional relation-aware attention
        e_i = rotate_model.entity_embedding[anchor_ids]  # [B,2d]
        q = self.w_q(e_i).unsqueeze(1)  # [B,1,A]
        k = self.w_k(hat)  # [B,K,A]
        logits = (q * k).sum(dim=-1) / max(self.attn_dim ** 0.5, 1e-6)  # [B,K]
        logits = logits.float()

        dir_id = a_dir.long()  # OUT=1, IN=0
        logits = logits + self.rel_bias(a_rel).squeeze(-1) + self.dir_bias(dir_id).squeeze(-1)

        logits = logits.masked_fill(~a_msk, -1e4)
        has_nbr = a_msk.any(dim=1)
        if (~has_nbr).any():
            logits[~has_nbr] = 0.0
        attn = torch.softmax(logits, dim=1)
        if (~has_nbr).any():
            attn[~has_nbr] = 0.0

        delta = torch.sum(attn.unsqueeze(-1) * hat, dim=1)  # [B,2d]
        if (~has_nbr).any():
            delta[~has_nbr] = e_i[~has_nbr]

        return delta, has_nbr, e_i

    def refine_anchor(
            self,
            anchor_ids: torch.LongTensor,  # [B]
            rotate_model,  # 传入 RotatE 实例以获取 Embeddings
            nbr_ent: torch.LongTensor,  # [N,K]
            nbr_rel: torch.LongTensor,  # [N,K]
            nbr_dir: torch.BoolTensor,  # [N,K]
            nbr_mask: torch.BoolTensor,  # [N,K]
            freq: torch.Tensor,  # [N]
            eta_floor: float | None = None,
    ) -> torch.Tensor:
        """
        返回 Refined 后的 Anchor Embedding [B, 2D]
        """
        delta, has_nbr, e_i = self._compute_ctx(
            anchor_ids, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask
        )

        # 6. 频次门控（Tail 更大注入）
        f = freq[anchor_ids].to(anchor_ids.device)  # [B]
        logf = torch.log1p(f)
        w = F.softplus(self.w_raw)
        eta = self.eta_max * torch.sigmoid(self.eta_raw + w * (-logf) + self.b)  # [B]
        eta = eta * has_nbr.float()
        if eta_floor is not None:
            eta = torch.clamp(eta, min=float(eta_floor))

        # 7. 计算残差并叠加
        re_delta, im_delta = torch.chunk(delta, 2, dim=-1)
        a = self.a.unsqueeze(0)  # [1,d]
        re_delta = a * re_delta
        im_delta = a * im_delta

        # 获取原始 Anchor Embedding
        re_i, im_i = torch.chunk(e_i, 2, dim=-1)

        re_out = re_i + eta.unsqueeze(-1) * (re_delta - re_i)
        im_out = im_i + eta.unsqueeze(-1) * (im_delta - im_i)

        return torch.cat([re_out, im_out], dim=-1)

    def score_delta_topk(
            self,
            anchor_ids: torch.LongTensor,  # [B]
            rel_ids: torch.LongTensor,     # [B]
            cand_ids: torch.LongTensor,    # [B,K] or [B]
            dir_ids: torch.LongTensor,     # [B] (RHS=0, LHS=1)
            rotate_model,
            nbr_ent: torch.LongTensor,
            nbr_rel: torch.LongTensor,
            nbr_dir: torch.BoolTensor,
            nbr_mask: torch.BoolTensor,
            freq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Candidate-aware delta scores for topK reranking.
        """
        if cand_ids.dim() == 1:
            cand_ids = cand_ids.unsqueeze(1)

        ctx, has_nbr, _ = self._compute_ctx(
            anchor_ids, rotate_model, nbr_ent, nbr_rel, nbr_dir, nbr_mask
        )
        q = self.q_proj(ctx)
        q = q + self.rel_bias_vec(rel_ids) + self.dir_bias_vec(dir_ids)

        e_cand = rotate_model.entity_embedding[cand_ids]  # [B,K,2d]
        v = self.t_proj(e_cand)  # [B,K,A]

        qf = q.float()
        vf = v.float()
        delta = torch.einsum("ba,bka->bk", qf, vf) / max(self.attn_dim ** 0.5, 1e-6)
        delta = delta.to(q.dtype)

        if (~has_nbr).any():
            delta = delta * has_nbr.float().unsqueeze(1)
        return delta