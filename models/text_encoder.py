import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os
from tqdm import tqdm


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', device='cuda'):
        super(TextEncoder, self).__init__()
        self.device = device
        # 这里的 pretrained_model 可以是本地路径，也可以是 huggingface ID
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(pretrained_model).to(device)
        self.bert.eval()

    def _encode_list(self, text_list, batch_size=64, max_len=64, description="Encoding"):
        """通用内部编码函数"""
        embs = []
        # 使用 tqdm 显示进度
        for i in tqdm(range(0, len(text_list), batch_size), desc=description):
            batch_texts = text_list[i: i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert(**inputs)
                # 使用 [CLS] token 作为句向量
                cls_emb = outputs.last_hidden_state[:, 0, :]
                embs.append(cls_emb.cpu())

        if len(embs) == 0:
            return torch.empty(0, 768)
        return torch.cat(embs, dim=0)

    def encode_all_entities(self, entity_text_list, save_path=None):
        # 实体描述通常较长，用 max_len=128 或 64
        embs = self._encode_list(entity_text_list, max_len=128, description="Encoding Entities")
        if save_path:
            torch.save(embs, save_path)
            print(f"[TextEncoder] Saved entity embeddings to {save_path}")
        return embs

    def encode_all_relations(self, relation_text_list, save_path=None):
        # 关系通常较短，用 max_len=64
        embs = self._encode_list(relation_text_list, max_len=64, description="Encoding Relations")
        if save_path:
            torch.save(embs, save_path)
            print(f"[TextEncoder] Saved relation embeddings to {save_path}")
        return embs