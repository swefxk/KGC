import torch
import torch.nn.functional as F


def compute_scale(top_scores_raw: torch.Tensor, std_floor: float = 1e-4) -> torch.Tensor:
    std = top_scores_raw.std(dim=1, unbiased=False)
    return std.clamp_min(std_floor)


def build_rank_features(
    top_scores: torch.Tensor,
    top_scores_raw: torch.Tensor,
    std_floor: float = 1e-4,
) -> torch.Tensor:
    """
    Build rank/score features for each candidate in Top-K.
    top_scores: [B,K] (union + sorted)
    top_scores_raw: [B,K] (raw topK before union)
    returns: [B,K,F]
    """
    B, K = top_scores.shape
    mean_raw = top_scores_raw.mean(dim=1)
    std_raw = top_scores_raw.std(dim=1, unbiased=False).clamp_min(std_floor)

    z = (top_scores - mean_raw.unsqueeze(1)) / std_raw.unsqueeze(1)
    gap_top1 = top_scores[:, :1] - top_scores
    if K > 1:
        gap_top2 = top_scores[:, 1:2] - top_scores
        gap_top2[:, 0] = 0.0
        margin12 = (top_scores[:, 0] - top_scores[:, 1]).unsqueeze(1)
    else:
        gap_top2 = torch.zeros_like(top_scores)
        margin12 = torch.zeros((B, 1), device=top_scores.device, dtype=top_scores.dtype)

    local_prev = torch.zeros_like(top_scores)
    local_prev[:, 1:] = top_scores[:, :-1] - top_scores[:, 1:]
    local_next = torch.zeros_like(top_scores)
    local_next[:, :-1] = top_scores[:, :-1] - top_scores[:, 1:]

    rank = (torch.arange(K, device=top_scores.device, dtype=top_scores.dtype) + 1.0) / float(K)
    rank = rank.unsqueeze(0).expand(B, -1)
    is_top10 = (rank * K <= 10).to(top_scores.dtype)
    is_top50 = (rank * K <= 50).to(top_scores.dtype)

    margin12 = margin12.expand(B, K)

    feats = torch.stack(
        [
            z,
            gap_top1,
            gap_top2,
            local_prev,
            local_next,
            rank,
            is_top10,
            is_top50,
            margin12,
        ],
        dim=-1,
    )
    return feats


def build_query_stats(
    top_scores: torch.Tensor,
    temp: float = 1.0,
    std_floor: float = 1e-4,
) -> torch.Tensor:
    """
    Query-level stats for the Top-K list.
    returns: [B,6] -> mean, std, entropy, m12, gap, range
    """
    mean = top_scores.mean(dim=1)
    std = top_scores.std(dim=1, unbiased=False).clamp_min(std_floor)
    p = F.softmax(top_scores / max(temp, 1e-6), dim=1)
    ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
    if top_scores.size(1) > 1:
        m12 = top_scores[:, 0] - top_scores[:, 1]
    else:
        m12 = torch.zeros_like(mean)
    gap = top_scores[:, 0] - mean
    rng = top_scores[:, 0] - top_scores[:, -1]
    return torch.stack([mean, std, ent, m12, gap, rng], dim=1)


def build_neighbor_features(
    anchor_ids: torch.Tensor,
    cand_ids: torch.Tensor,
    rel_ids: torch.Tensor,
    dir_val: int,
    graph_stats: dict,
):
    """
    Build 1-hop neighborhood features (backbone-agnostic).
    returns:
      nbr_feats: [B,K,F]
      anchor_hist_ids/counts: [B,M]
      cand_hist_ids/counts: [B,K,M]
    """
    device = anchor_ids.device
    nbr_ent = graph_stats["nbr_ent"]
    nbr_rel = graph_stats["nbr_rel"]
    nbr_dir = graph_stats["nbr_dir"]
    nbr_mask = graph_stats["nbr_mask"]
    deg = graph_stats["deg"]
    freq = graph_stats["freq"]
    rel_hist_ids = graph_stats["rel_hist_ids"]
    rel_hist_counts = graph_stats["rel_hist_counts"]

    B, K = cand_ids.shape

    anchor_nbr_ent = nbr_ent[anchor_ids]        # [B,Kn]
    anchor_nbr_rel = nbr_rel[anchor_ids]
    anchor_nbr_dir = nbr_dir[anchor_ids]
    anchor_nbr_mask = nbr_mask[anchor_ids]
    anchor_has_nbr = anchor_nbr_mask.any(dim=1).to(torch.float)

    cand_nbr_ent = nbr_ent[cand_ids]           # [B,K,Kn]
    cand_nbr_rel = nbr_rel[cand_ids]
    cand_nbr_dir = nbr_dir[cand_ids]
    cand_nbr_mask = nbr_mask[cand_ids]
    cand_has_nbr = cand_nbr_mask.any(dim=2).to(torch.float)

    # cand in anchor 1-hop
    cand_in_anchor = (cand_ids.unsqueeze(-1) == anchor_nbr_ent.unsqueeze(1)) & anchor_nbr_mask.unsqueeze(1)
    cand_in_anchor = cand_in_anchor.any(dim=-1).to(torch.float)

    # relation match count (anchor)
    dir_val_t = torch.tensor(dir_val, device=device, dtype=anchor_nbr_dir.dtype)
    rel_match_mask = (anchor_nbr_rel == rel_ids.unsqueeze(1)) & (anchor_nbr_dir == dir_val_t) & anchor_nbr_mask
    rel_match_cnt = rel_match_mask.sum(dim=1).to(torch.float)
    rel_match_cnt = rel_match_cnt.unsqueeze(1).expand(B, K)

    # candidate in anchor with relation r and dir
    cand_in_anchor_rel = (cand_ids.unsqueeze(-1) == anchor_nbr_ent.unsqueeze(1)) & rel_match_mask.unsqueeze(1)
    cand_in_anchor_rel = cand_in_anchor_rel.any(dim=-1).to(torch.float)

    # common neighbors (truncated)
    anchor_nbr_ent_masked = anchor_nbr_ent.clone()
    anchor_nbr_ent_masked[~anchor_nbr_mask] = -1
    cand_nbr_ent_masked = cand_nbr_ent.clone()
    cand_nbr_ent_masked[~cand_nbr_mask] = -1
    eq = cand_nbr_ent_masked.unsqueeze(-1) == anchor_nbr_ent_masked.unsqueeze(1).unsqueeze(2)
    match = eq.any(dim=-1)
    match = match & cand_nbr_mask
    common_nbr = match.sum(dim=-1).to(torch.float)

    deg_anchor = torch.log1p(deg[anchor_ids]).unsqueeze(1).expand(B, K)
    deg_cand = torch.log1p(deg[cand_ids])
    freq_cand = torch.log1p(freq[cand_ids])
    common_nbr = torch.log1p(common_nbr)
    rel_match_cnt = torch.log1p(rel_match_cnt)

    nbr_feats = torch.stack(
        [
            deg_anchor,
            deg_cand,
            common_nbr,
            rel_match_cnt,
            cand_in_anchor,
            cand_in_anchor_rel,
            freq_cand,
            anchor_has_nbr.unsqueeze(1).expand(B, K),
            cand_has_nbr,
        ],
        dim=-1,
    )

    anchor_hist_ids = rel_hist_ids[anchor_ids]
    anchor_hist_counts = rel_hist_counts[anchor_ids]
    cand_hist_ids = rel_hist_ids[cand_ids]
    cand_hist_counts = rel_hist_counts[cand_ids]

    return nbr_feats, anchor_hist_ids, anchor_hist_counts, cand_hist_ids, cand_hist_counts
