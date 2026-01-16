import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticBiEncoderScorer(nn.Module):
    """
    SimKGC-style bi-encoder scorer on frozen text embeddings.

    forward(h_txt, r_txt, t_txt, r_ids) -> (delta, lam) with lam=1 so sem_score=delta.
    Also exposes encode_query / encode_entity for fast matrix-mul eval.
    """
    def __init__(
        self,
        text_dim: int,
        num_relations: int,
        proj_dim: int = 256,
        hidden_mult: int = 2,
        dropout: float = 0.1,
        text_norm: bool = True,
    ):
        super().__init__()
        self.text_dim = int(text_dim)
        self.num_relations = int(num_relations)
        self.proj_dim = int(proj_dim)
        self.text_norm = bool(text_norm)

        h = max(256, proj_dim * hidden_mult)

        self.ent_proj = MLP(self.text_dim, self.proj_dim, h, dropout)
        self.rel_proj = MLP(self.text_dim, self.proj_dim, h, dropout)
        self.dir_emb = nn.Embedding(2, self.text_dim)
        nn.init.zeros_(self.dir_emb.weight)

        self.q_fuse = nn.Sequential(
            nn.Linear(self.proj_dim * 4, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
        )

    def _norm_in(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1) if self.text_norm else x

    def encode_entity(self, e_txt: torch.Tensor) -> torch.Tensor:
        e = self._norm_in(e_txt)
        v = self.ent_proj(e)
        return F.normalize(v, p=2, dim=-1)

    def encode_relation(self, r_txt: torch.Tensor) -> torch.Tensor:
        r = self._norm_in(r_txt)
        v = self.rel_proj(r)
        return F.normalize(v, p=2, dim=-1)

    def encode_query(self, anchor_txt: torch.Tensor, rel_txt: torch.Tensor, dir_ids: torch.Tensor = None) -> torch.Tensor:
        a = self.encode_entity(anchor_txt)
        if dir_ids is None:
            r_txt = rel_txt
        else:
            r_txt = rel_txt + self.dir_emb(dir_ids.to(rel_txt.device))
        r = self.encode_relation(r_txt)
        feat = torch.cat([a, r, a * r, a - r], dim=-1)
        q = self.q_fuse(feat)
        return F.normalize(q, p=2, dim=-1)

    def forward(
        self,
        h_txt: torch.Tensor,
        r_txt: torch.Tensor,
        t_txt: torch.Tensor,
        r_ids: torch.Tensor = None,
        dir_ids: torch.Tensor = None,
    ):
        q = self.encode_query(h_txt, r_txt, dir_ids=dir_ids)
        v = self.encode_entity(t_txt)
        delta = (q * v).sum(dim=-1)
        lam = torch.ones_like(delta)
        return delta, lam
