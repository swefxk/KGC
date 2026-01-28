import torch
import torch.nn as nn

from models.struct_backbone_base import StructBackboneBase


class ComplExBackbone(StructBackboneBase):
    """
    ComplEx structural backbone with entity/relation embeddings in (re, im).
    entity_embedding: [N, 2d]
    relation_embedding: [R, 2d]
    """

    def __init__(self, num_entities: int, num_relations: int, emb_dim: int):
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.emb_dim = int(emb_dim)

        self._entity_embedding = nn.Parameter(torch.zeros(self.num_entities, self.emb_dim * 2))
        self._relation_embedding = nn.Parameter(torch.zeros(self.num_relations, self.emb_dim * 2))

        nn.init.xavier_uniform_(self._entity_embedding)
        nn.init.xavier_uniform_(self._relation_embedding)

    @property
    def entity_embedding(self) -> torch.Tensor:
        return self._entity_embedding

    @property
    def relation_embedding(self) -> torch.Tensor:
        return self._relation_embedding

    def get_relation_reim(self, r_idx, conj=False):
        re_r, im_r = torch.chunk(self._relation_embedding[r_idx], 2, dim=-1)
        if isinstance(conj, torch.Tensor):
            conj = conj.to(im_r.device).bool()
            mask = conj.unsqueeze(-1)
            im_r = torch.where(mask, -im_r, im_r)
        elif conj:
            im_r = -im_r
        return re_r, im_r

    def _looks_like_contiguous_candidates(self, t_ids: torch.Tensor) -> bool:
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
        Compatible with RotatE score_from_head_emb signature.
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
            t_emb = self._entity_embedding[t_ids].unsqueeze(0)  # [1, C, 2d]
            re_hr = re_hr.unsqueeze(1)
            im_hr = im_hr.unsqueeze(1)
        elif t_ids.dim() == 2:
            dim0 = t_ids.size(0)
            if dim0 != 1 and dim0 != B:
                raise ValueError(
                    f"Invalid 2D t_ids shape {t_ids.shape}. First dimension must be 1 or B={B}."
                )
            t_emb = self._entity_embedding[t_ids]  # [1/B, K, 2d]
            re_hr = re_hr.unsqueeze(1)
            im_hr = im_hr.unsqueeze(1)
        else:
            raise ValueError(f"t_ids must be 1D [C] or 2D [1/B, K]. Got {t_ids.shape}.")

        re_t, im_t = torch.chunk(t_emb, 2, dim=-1)
        score = (re_hr * re_t + im_hr * im_t).sum(dim=-1)
        return score

    def score_tail_batch(self, h_ids, r_ids, cand_t_ids):
        h_emb = self._entity_embedding[h_ids]
        conj_flag = torch.zeros_like(r_ids, dtype=torch.bool, device=h_emb.device)
        return self.score_from_head_emb(h_emb, r_ids, cand_t_ids, conj=conj_flag)

    def score_head_batch(self, r_ids, t_ids, cand_h_ids):
        t_emb = self._entity_embedding[t_ids]
        conj_flag = torch.ones_like(r_ids, dtype=torch.bool, device=t_emb.device)
        return self.score_from_head_emb(t_emb, r_ids, cand_h_ids, conj=conj_flag)

    def forward(self, h, r, t, mode="single"):
        h_emb = self._entity_embedding[h]
        if mode == "single":
            return self.score_from_head_emb(h_emb, r, t.unsqueeze(1)).squeeze(1)
        if mode in ("batch_neg", "tail_predict"):
            return self.score_from_head_emb(h_emb, r, t)
        raise ValueError(f"Invalid mode: {mode}")

    def score_rhs_all(self, h_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
        all_ent = torch.arange(self.num_entities, device=h_ids.device, dtype=torch.long)
        return self.score_tail_batch(h_ids, r_ids, all_ent)

    def score_lhs_all(self, t_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
        all_ent = torch.arange(self.num_entities, device=t_ids.device, dtype=torch.long)
        return self.score_head_batch(r_ids, t_ids, all_ent)

    def score_rhs_cands(self, h_ids: torch.Tensor, r_ids: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        return self.score_tail_batch(h_ids, r_ids, cand_ids)

    def score_lhs_cands(self, t_ids: torch.Tensor, r_ids: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        return self.score_head_batch(r_ids, t_ids, cand_ids)
