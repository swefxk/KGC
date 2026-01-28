import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class StructBackboneBase(nn.Module, ABC):
    """
    Minimal interface for structural backbones used by topK-inject pipeline.
    """

    @property
    @abstractmethod
    def entity_embedding(self) -> torch.Tensor:
        """Entity embedding tensor with shape [num_entities, 2d]."""

    @abstractmethod
    def score_rhs_all(self, h_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
        """Score all tail entities for (h, r). Return [B, num_entities]."""

    @abstractmethod
    def score_lhs_all(self, t_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
        """Score all head entities for (t, r). Return [B, num_entities]."""

    @abstractmethod
    def score_rhs_cands(
        self,
        h_ids: torch.Tensor,
        r_ids: torch.Tensor,
        cand_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate tails. Return [B, K] or [B, C]."""

    @abstractmethod
    def score_lhs_cands(
        self,
        t_ids: torch.Tensor,
        r_ids: torch.Tensor,
        cand_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate heads. Return [B, K] or [B, C]."""
