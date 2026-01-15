import torch
import torch.nn as nn


class RotatEModel(nn.Module):
    def __init__(self, num_entities, num_relations, emb_dim, margin=9.0, epsilon=2.0):
        super(RotatEModel, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.margin = margin
        self.epsilon = epsilon

        self.embedding_range = (margin + epsilon) / emb_dim

        # entity: [N, 2D]
        self.entity_embedding = nn.Parameter(torch.zeros(num_entities, emb_dim * 2))
        nn.init.uniform_(self.entity_embedding, -self.embedding_range, self.embedding_range)

        # relation: [R, D] (phase)
        self.relation_embedding = nn.Parameter(torch.zeros(num_relations, emb_dim))
        nn.init.uniform_(self.relation_embedding, -self.embedding_range, self.embedding_range)

    def get_relation_reim(self, r_idx, conj=False):
        """
        支持 conj 张量广播
        """
        phase_r = self.relation_embedding[r_idx] / (self.embedding_range / 3.141592653589793)
        re_r = torch.cos(phase_r)
        im_r = torch.sin(phase_r)

        if isinstance(conj, torch.Tensor):
            conj = conj.to(im_r.device).bool()
            mask = conj.unsqueeze(-1)
            im_r = torch.where(mask, -im_r, im_r)
        elif conj:
            im_r = -im_r

        return re_r, im_r

    def _looks_like_contiguous_candidates(self, t_ids):
        """
        判断 1D tensor 是否严格符合 Contiguous Candidate Chunk：
        必须是 arange 切片 (连续、递增、不越界)。
        """
        if t_ids.numel() == 0:
            return True
        if t_ids.numel() == 1:
            val = t_ids[0].item()
            return 0 <= val < self.num_entities

        first = t_ids[0].item()
        last = t_ids[-1].item()
        if first < 0 or last >= self.num_entities:
            return False

        return torch.all(t_ids[1:] == t_ids[:-1] + 1).item()

    def score_from_head_emb(self, h_emb, r_idx, t_ids, conj=False):
        """
        接口契约：
        - 1D [C]: 必须是 Contiguous Candidate Chunk。否则报错。
        - 2D [1, K] or [B, K]: Shared Candidates 或 Per-Sample Candidates。
        """
        if t_ids.dtype != torch.long:
            raise TypeError(f"t_ids must be torch.long, got {t_ids.dtype}")

        re_h, im_h = torch.chunk(h_emb, 2, dim=-1)
        re_r, im_r = self.get_relation_reim(r_idx, conj=conj)

        re_hr = re_h * re_r - im_h * im_r
        im_hr = re_h * im_r + im_h * re_r

        B = h_emb.size(0)

        if t_ids.dim() == 1:
            if not self._looks_like_contiguous_candidates(t_ids):
                raise ValueError(
                    "Invalid 1D t_ids input. In this project, 1D t_ids implies a Contiguous Candidate Chunk.\n"
                    f"Got non-contiguous, out-of-bound, or device-mismatched data. shape={t_ids.shape}.\n"
                    "For non-contiguous candidates, reshape to [1, K] (shared) or [B, K] (per-sample)."
                )

            t_emb = self.entity_embedding[t_ids].unsqueeze(0)  # [1, C, 2D]
            re_hr = re_hr.unsqueeze(1)  # [B, 1, D]
            im_hr = im_hr.unsqueeze(1)

        elif t_ids.dim() == 2:
            dim0 = t_ids.size(0)
            if dim0 != 1 and dim0 != B:
                raise ValueError(
                    f"Invalid 2D t_ids shape {t_ids.shape}. First dimension must be 1 or B={B}."
                )

            t_emb = self.entity_embedding[t_ids]  # [1/B, K, 2D]
            re_hr = re_hr.unsqueeze(1)
            im_hr = im_hr.unsqueeze(1)
        else:
            raise ValueError(f"t_ids must be 1D [C] or 2D [1/B, K]. Got {t_ids.shape}.")

        re_t, im_t = torch.chunk(t_emb, 2, dim=-1)

        re_score = re_hr - re_t
        im_score = im_hr - im_t

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)      # L2 over (re,im)
        score = score.sum(dim=-1)      # sum over dims

        return self.margin - score

    # --- 新增：不破坏现有用户代码的辅助接口 ---
    def score_tail_batch(self, h_ids, r_ids, cand_t_ids):
        """
        给定 (h,r) 打分候选 t：
        cand_t_ids: [B,K] 或 [1,K] 或 [C] contiguous chunk
        """
        h_emb = self.entity_embedding[h_ids]
        conj_flag = torch.zeros_like(r_ids, dtype=torch.bool, device=h_emb.device)
        return self.score_from_head_emb(h_emb, r_ids, cand_t_ids, conj=conj_flag)

    def score_head_batch(self, r_ids, t_ids, cand_h_ids):
        """
        给定 (r,t) 打分候选 h，使用等价变换：
        score(h,r,t) == score(t, conj(r), h)
        cand_h_ids: [B,K] / [1,K] / contiguous chunk
        """
        t_emb = self.entity_embedding[t_ids]
        conj_flag = torch.ones_like(r_ids, dtype=torch.bool, device=t_emb.device)
        return self.score_from_head_emb(t_emb, r_ids, cand_h_ids, conj=conj_flag)

    def forward(self, h, r, t, mode='single'):
        h_emb = self.entity_embedding[h]
        if mode == 'single':
            return self.score_from_head_emb(h_emb, r, t.unsqueeze(1)).squeeze(1)
        elif mode == 'batch_neg' or mode == 'tail_predict':
            return self.score_from_head_emb(h_emb, r, t)
        else:
            raise ValueError("Invalid mode")
