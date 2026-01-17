import torch
import torch.nn.functional as F
import numpy as np


class MultiViewNegativeSampler:
    def __init__(self, num_entities, text_embeddings,
                 one_hop_neighbors: dict,
                 num_candidates=64, num_negatives=16,
                 alpha=0.1,
                 beta=0.0,
                 gamma=0.9,
                 temperature=1.0,
                 device='cuda'):
        """
        三视角负采样器
        """
        self.num_entities = num_entities
        self.text_embeddings = text_embeddings.to(device)
        self.text_embeddings = F.normalize(self.text_embeddings, p=2, dim=1)
        self.one_hop_neighbors = one_hop_neighbors
        self.num_candidates = num_candidates
        self.num_negatives = num_negatives
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.device = device

    def sample(self, rotatE_model, head, relation, true_tail):
        batch_size = head.size(0)

        # 1. 候选生成
        cand_tails = torch.randint(0, self.num_entities, (batch_size, self.num_candidates), device=self.device)

        # 2. 语义视角
        true_tail_emb = self.text_embeddings[true_tail].unsqueeze(1)
        cand_emb = self.text_embeddings[cand_tails]
        cosine_sim = (true_tail_emb * cand_emb).sum(dim=-1)
        h_sem = (cosine_sim + 1.0) / 2.0

        # 3. 结构视角 (暂时关闭 beta=0)
        h_struct = 0.0

        # 4. 模型视角
        with torch.no_grad():
            model_score = rotatE_model(head, relation, cand_tails, mode='batch_neg')
        h_model = torch.sigmoid(model_score)

        # 5. 融合
        H = (self.alpha * h_sem) + (self.beta * h_struct) + (self.gamma * h_model)

        # 6. 采样
        probs = F.softmax(H / self.temperature, dim=1)
        neg_indices = torch.multinomial(probs, self.num_negatives, replacement=True)
        final_neg_tails = torch.gather(cand_tails, 1, neg_indices)

        return final_neg_tails