import torch
import torch.nn as nn


class RankStructRefiner(nn.Module):
    """
    Backbone-agnostic rank-and-structure refiner (Δ-X).
    Input: rank/score features + 1-hop neighborhood stats + rel/dir embeddings + query stats.
    Output: delta scores for topK candidates.
    """

    def __init__(
        self,
        num_relations: int,
        rank_feat_dim: int,
        nbr_feat_dim: int,
        q_stat_dim: int,
        rel_emb_dim: int = 64,
        dir_emb_dim: int = 16,
        model_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        delta_cap: float = 4.0,
        max_pos: int = 500,
        use_hist: bool = True,
    ):
        super().__init__()
        self.num_relations = int(num_relations)
        self.rank_feat_dim = int(rank_feat_dim)
        self.nbr_feat_dim = int(nbr_feat_dim)
        self.q_stat_dim = int(q_stat_dim)
        self.rel_emb_dim = int(rel_emb_dim)
        self.dir_emb_dim = int(dir_emb_dim)
        self.model_dim = int(model_dim)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.delta_cap = float(delta_cap)
        self.max_pos = int(max_pos)
        self.use_hist = bool(use_hist)

        self.rel_emb = nn.Embedding(self.num_relations, self.rel_emb_dim)
        self.dir_emb = nn.Embedding(2, self.dir_emb_dim)

        hist_dim = self.rel_emb_dim * 2 if self.use_hist else 0
        in_dim = (
            self.rank_feat_dim
            + self.nbr_feat_dim
            + self.q_stat_dim
            + self.rel_emb_dim
            + self.dir_emb_dim
            + hist_dim
        )

        self.proj_in = nn.Linear(in_dim, self.model_dim)
        self.pos_emb = nn.Embedding(self.max_pos, self.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.n_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.out = nn.Linear(self.model_dim, 1)

    def _hist_embed(self, rel_ids: torch.Tensor, rel_counts: torch.Tensor) -> torch.Tensor:
        """
        rel_ids: [B,M] or [B,K,M] with -1 padding
        rel_counts: same shape (float)
        return: [B, D] or [B,K,D]
        """
        if rel_ids is None or rel_counts is None:
            raise ValueError("rel_ids and rel_counts required when use_hist=True")
        mask = rel_ids >= 0
        rel_ids_safe = rel_ids.clamp_min(0)
        emb = self.rel_emb(rel_ids_safe)
        weights = rel_counts * mask.to(rel_counts.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return (emb * weights.unsqueeze(-1)).sum(dim=-2)

    def score_delta_topk_rank(
        self,
        rank_feats: torch.Tensor,    # [B,K,rank_feat_dim]
        nbr_feats: torch.Tensor,     # [B,K,nbr_feat_dim]
        q_stats: torch.Tensor,       # [B,q_stat_dim]
        rel_ids: torch.Tensor,       # [B]
        dir_ids: torch.Tensor,       # [B]
        scale: torch.Tensor | None,  # [B]
        anchor_hist_ids: torch.Tensor | None = None,   # [B,M]
        anchor_hist_counts: torch.Tensor | None = None,  # [B,M]
        cand_hist_ids: torch.Tensor | None = None,     # [B,K,M]
        cand_hist_counts: torch.Tensor | None = None,  # [B,K,M]
    ) -> torch.Tensor:
        B, K, _ = rank_feats.shape
        if K > self.max_pos:
            raise ValueError(f"topK={K} exceeds max_pos={self.max_pos}")

        rel = self.rel_emb(rel_ids).unsqueeze(1).expand(B, K, -1)
        dire = self.dir_emb(dir_ids).unsqueeze(1).expand(B, K, -1)
        q = q_stats.unsqueeze(1).expand(B, K, -1)

        parts = [rank_feats, nbr_feats, q, rel, dire]
        if self.use_hist:
            anchor_hist = self._hist_embed(anchor_hist_ids, anchor_hist_counts)  # [B,D]
            cand_hist = self._hist_embed(cand_hist_ids, cand_hist_counts)        # [B,K,D]
            anchor_hist = anchor_hist.unsqueeze(1).expand(B, K, -1)
            parts.extend([anchor_hist, cand_hist])

        x = torch.cat(parts, dim=-1)
        x = self.proj_in(x)
        pos = self.pos_emb(torch.arange(K, device=x.device))
        x = x + pos.unsqueeze(0)
        x = self.encoder(x)
        u = torch.tanh(self.out(x).squeeze(-1)) * self.delta_cap
        if scale is not None:
            u = u * scale.unsqueeze(1)
        return u
