import torch

from models.rotate import RotatEModel
from models.struct_backbone_base import StructBackboneBase


class RotatEBackbone(StructBackboneBase):
    """
    Thin wrapper over RotatEModel to satisfy StructBackboneBase interface.
    """

    def __init__(self, num_entities: int, num_relations: int, emb_dim: int, margin: float = 9.0, epsilon: float = 2.0):
        super().__init__()
        self.model = RotatEModel(
            num_entities=num_entities,
            num_relations=num_relations,
            emb_dim=emb_dim,
            margin=margin,
            epsilon=epsilon,
        )

    @property
    def entity_embedding(self) -> torch.Tensor:
        return self.model.entity_embedding

    @property
    def num_entities(self) -> int:
        return self.model.num_entities

    @property
    def num_relations(self) -> int:
        return self.model.num_relations

    def get_relation_reim(self, r_idx, conj=False):
        return self.model.get_relation_reim(r_idx, conj=conj)

    def score_from_head_emb(self, h_emb, r_idx, t_ids, conj=False):
        return self.model.score_from_head_emb(h_emb, r_idx, t_ids, conj=conj)

    def score_tail_batch(self, h_ids, r_ids, cand_t_ids):
        return self.model.score_tail_batch(h_ids, r_ids, cand_t_ids)

    def score_head_batch(self, r_ids, t_ids, cand_h_ids):
        return self.model.score_head_batch(r_ids, t_ids, cand_h_ids)

    def forward(self, h, r, t, mode="single"):
        return self.model(h, r, t, mode=mode)

    def score_rhs_all(self, h_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
        all_ent = torch.arange(self.model.num_entities, device=h_ids.device, dtype=torch.long)
        h_emb = self.model.entity_embedding[h_ids]
        conj_flag = torch.zeros_like(r_ids, dtype=torch.bool, device=h_ids.device)
        return self.model.score_from_head_emb(h_emb, r_ids, all_ent, conj=conj_flag)

    def score_lhs_all(self, t_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
        all_ent = torch.arange(self.model.num_entities, device=t_ids.device, dtype=torch.long)
        t_emb = self.model.entity_embedding[t_ids]
        conj_flag = torch.ones_like(r_ids, dtype=torch.bool, device=t_ids.device)
        return self.model.score_from_head_emb(t_emb, r_ids, all_ent, conj=conj_flag)

    def score_rhs_cands(self, h_ids: torch.Tensor, r_ids: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        h_emb = self.model.entity_embedding[h_ids]
        conj_flag = torch.zeros_like(r_ids, dtype=torch.bool, device=h_ids.device)
        return self.model.score_from_head_emb(h_emb, r_ids, cand_ids, conj=conj_flag)

    def score_lhs_cands(self, t_ids: torch.Tensor, r_ids: torch.Tensor, cand_ids: torch.Tensor) -> torch.Tensor:
        t_emb = self.model.entity_embedding[t_ids]
        conj_flag = torch.ones_like(r_ids, dtype=torch.bool, device=t_ids.device)
        return self.model.score_from_head_emb(t_emb, r_ids, cand_ids, conj=conj_flag)
