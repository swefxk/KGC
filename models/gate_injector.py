import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_from_logits(x: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    z = x / max(temp, 1e-6)
    p = F.softmax(z, dim=1)
    return -(p * (p.clamp_min(1e-12).log())).sum(dim=1)


def gate_features_from_top_scores(top_scores: torch.Tensor, ent_temp: float = 1.0) -> torch.Tensor:
    """
    top_scores: [B,K] already topK-sorted (desc)
    return: feats [B,5]
    """
    s1 = top_scores[:, 0]
    s2 = top_scores[:, 1] if top_scores.size(1) > 1 else top_scores[:, 0]
    m12 = s1 - s2
    mean = top_scores.mean(dim=1)
    std = top_scores.std(dim=1, unbiased=False)
    gap = s1 - mean
    ent = entropy_from_logits(top_scores, temp=ent_temp)
    return torch.stack([m12, mean, std, gap, ent], dim=1)


class ConfidenceGate(nn.Module):
    """
    g(q) = g_max * sigmoid( MLP([stats(top_scores), emb(r), emb(dir)]) )
    """
    def __init__(
        self,
        num_relations: int,
        rel_emb_dim: int = 16,
        dir_emb_dim: int = 8,
        hidden_dim: int = 64,
        g_min: float = 0.0,
        g_max: float = 2.0,
        init_bias: float = 0.5413,  # softplus(init_bias) ~= 1.0
    ):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, rel_emb_dim)
        self.dir_emb = nn.Embedding(2, dir_emb_dim)
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.init_bias = float(init_bias)

        in_dim = 5 + rel_emb_dim + dir_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # make gate start "almost off"
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, top_scores: torch.Tensor, r_ids: torch.Tensor, dir_ids: torch.Tensor, ent_temp: float = 1.0):
        feats = gate_features_from_top_scores(top_scores, ent_temp=ent_temp)  # [B,5]
        x = torch.cat([feats, self.rel_emb(r_ids), self.dir_emb(dir_ids)], dim=1)
        u = self.mlp(x).squeeze(1)
        g = F.softplus(u + self.init_bias)
        return torch.clamp(g, min=self.g_min, max=self.g_max)
