import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticResidualScorerV3(nn.Module):
    """
    V3: stable, zero-start, relation-calibrated semantic residual.

    Output interface is kept compatible with existing code:
        delta, lam = model(h_txt, r_txt, t_txt, r_ids)
        sem_score = lam * delta
    """
    def __init__(
        self,
        text_dim: int,
        num_relations: int,
        lambda_base: float = 0.01,
        text_norm: bool = True,
        delta_bound: float = 0.0,
        hidden_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.num_relations = num_relations
        self.lambda_base = float(lambda_base)
        self.text_norm = bool(text_norm)
        self.delta_bound = float(delta_bound)

        hdim = max(64, text_dim * hidden_mult)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim * 2, hdim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hdim, text_dim),
        )

        # Relation-specific scale. IMPORTANT: init=0 => lam=0 => sem_score=0 at start.
        self.rel_alpha = nn.Embedding(num_relations, 1)
        nn.init.zeros_(self.rel_alpha.weight)

    def forward(self, h_txt: torch.Tensor, r_txt: torch.Tensor, t_txt: torch.Tensor, r_ids: torch.Tensor):
        """
        h_txt, r_txt, t_txt: [N, D]
        r_ids: [N]
        """
        if self.text_norm:
            h_txt = F.normalize(h_txt, p=2, dim=-1)
            r_txt = F.normalize(r_txt, p=2, dim=-1)
            t_txt = F.normalize(t_txt, p=2, dim=-1)

        q = self.mlp(torch.cat([h_txt, r_txt], dim=-1))  # [N, D]
        # scaled dot-product
        delta = (q * t_txt).sum(dim=-1) / math.sqrt(self.text_dim)  # [N]

        if self.delta_bound > 0:
            delta = torch.clamp(delta, -self.delta_bound, self.delta_bound)

        alpha = self.rel_alpha(r_ids).squeeze(-1)  # [N]
        lam = self.lambda_base * torch.tanh(alpha)  # [-lambda_base, +lambda_base]

        return delta, lam
