import torch
import torch.nn as nn
import torch.nn.functional as F


def _sem_stats(s_sem: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    mean = s_sem.mean(dim=1)
    std = s_sem.std(dim=1, unbiased=False)
    maxv = s_sem.max(dim=1).values
    gap = maxv - mean
    p = F.softmax(s_sem / max(temp, 1e-6), dim=1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
    return torch.stack([mean, std, maxv, gap, ent], dim=1)


def _topk_agree(s_sem: torch.Tensor, s_struct: torch.Tensor, topm: int) -> torch.Tensor:
    k = max(1, min(int(topm), s_sem.size(1)))
    sem_idx = torch.topk(s_sem, k=k, dim=1).indices  # [B,k]
    struct_idx = torch.topk(s_struct, k=k, dim=1).indices  # [B,k]
    match = (sem_idx.unsqueeze(2) == struct_idx.unsqueeze(1)).any(dim=2).float().sum(dim=1)
    return match / float(k)


class SemanticConfidenceNet(nn.Module):
    """
    Query-level semantic reliability r(q) in [r_min, r_max].
    Inputs are sem/struct topK scores + relation/direction ids.
    """

    def __init__(
        self,
        num_relations: int,
        rel_emb_dim: int = 32,
        dir_emb_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        r_min: float = 0.05,
        r_max: float = 0.95,
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)
        self.dir_emb = nn.Embedding(2, dir_emb_dim)
        self.r_min = float(r_min)
        self.r_max = float(r_max)

        in_dim = 6 + rel_emb_dim + dir_emb_dim  # sem_stats(5) + agree(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        s_sem: torch.Tensor,    # [B,K]
        s_struct: torch.Tensor, # [B,K]
        rel_ids: torch.Tensor,  # [B]
        dir_ids: torch.Tensor,  # [B]
        topm: int = 10,
        temp: float = 1.0,
    ) -> torch.Tensor:
        stats = _sem_stats(s_sem, temp=temp)
        agree = _topk_agree(s_sem, s_struct, topm=topm).unsqueeze(1)
        x = torch.cat([stats, agree, self.rel_emb(rel_ids), self.dir_emb(dir_ids)], dim=1)
        r = torch.sigmoid(self.mlp(x)).squeeze(1)
        return torch.clamp(r, min=self.r_min, max=self.r_max)
