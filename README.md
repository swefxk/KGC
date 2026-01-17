# KGC Project

## Mainline Minimal Entry Points

Build:
- `build/build_rotate_cache_rhs_topk.py`
- `build/build_rotate_cache_lhs_topk.py`

Train:
- `train/train_rotate.py`
- `train/train_sem_biencoder.py`
- `train/train_gate_inject_topk.py`
- `train/train_struct_refiner.py`

Eval:
- `eval/eval_full_entity_filtered.py`
- `eval/eval_topk_inject.py`

## Minimal Reproduction (Mainline)

## Mainline Entry Points (Minimal Set)

| Stage | Script | Purpose | Output / Artifact |
|---|---|---|---|
| Build | `build/build_rotate_cache_rhs_topk.py` | Build RHS topK hard-negative cache (filtered) | `data/<ds>/cache/train_rhs_top*_*.pt` |
| Build | `build/build_rotate_cache_lhs_topk.py` | Build LHS topK hard-negative cache (filtered) | `data/<ds>/cache/train_lhs_top*_*.pt` |
| Train | `train/train_rotate.py` | Train structural retriever (RotatE) | `checkpoints/.../rotate/best_model.pth` |
| Train | `train/train_sem_biencoder.py` | Train semantic bi-encoder (RHS+LHS) | `checkpoints/.../sem_biencoder/biencoder_best.pth` |
| Train | `train/train_gate_inject_topk.py` | Train adaptive injection gate (topK-only aligned) | `checkpoints/.../gate/gate_best.pth` |
| Train (opt) | `train/train_struct_refiner.py` | Train structural refiner / reranker | `checkpoints/.../refiner/refiner_best.pth` |
| Eval | `eval/eval_full_entity_filtered.py` | Full-entity filtered evaluation (truth source) | Printed metrics + optional logs |
| Eval | `eval/eval_topk_inject.py` | Final system eval (topK-only injection + gate + Δ + CI) | Printed metrics + paired bootstrap CI |

Mainline default reports AVG (RHS+LHS)/2 with RHS/LHS splits, Hits@{1,3,10}, Recall@K, and paired bootstrap ΔMRR CI.

### Requirements
- Python 3.10+
- PyTorch (CUDA recommended)
- Dataset under `data/<dataset_name>/`:
  - `train.txt`, `valid.txt`, `test.txt`
  - `entities.dict`, `relations.dict`
  - `text_embs.pt`, `rel_text_embs.pt`
  - `cache/` will be created

### One-command run
```bash
bash scripts/run_mainline_full.sh fb15k_custom
```

### Output artifacts
- RotatE: `checkpoints/mainline_<dataset>/rotate/best_model.pth`
- Bi-encoder: `checkpoints/mainline_<dataset>/sem_biencoder/biencoder_best.pth`
- Gate: `checkpoints/mainline_<dataset>/gate/gate_best.pth`
- Refiner: `checkpoints/mainline_<dataset>/refiner/refiner_best.pth`
- Final eval printed by `eval/eval_topk_inject.py`:
  - Filtered MRR / Hits@{1,3,10} for RHS/LHS/AVG
  - Bucket metrics (Head/Torso/Tail)
  - Recall@K (gold in topK)
  - Paired bootstrap CI for ΔMRR (when enabled)

### Mainline Default Hyperparameters
- `topK=200`
- `b_rhs=2.0`
- `b_lhs=2.5`
- `gamma_rhs=2.0`
- `gamma_lhs=0.0`
- Safety variant: `delta_gate_ent_q=0.6`
