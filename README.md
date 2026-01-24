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
| Build | `build/build_rotate_cache_rhs_topk.py` | Build RHS topK hard-negative cache (filtered) | `artifacts/<run_id>/cache/train_rhs_top*_*.pt` |
| Build | `build/build_rotate_cache_lhs_topk.py` | Build LHS topK hard-negative cache (filtered) | `artifacts/<run_id>/cache/train_lhs_top*_*.pt` |
| Train | `train/train_rotate.py` | Train structural retriever (RotatE) | `artifacts/<run_id>/checkpoints/rotate/best_model.pth` |
| Train | `train/train_sem_biencoder.py` | Train semantic bi-encoder (RHS+LHS) | `artifacts/<run_id>/checkpoints/sem_biencoder/biencoder_best.pth` |
| Train | `train/train_gate_inject_topk.py` | Train adaptive injection gate (topK-only aligned) | `artifacts/<run_id>/checkpoints/gate/gate_best.pth` |
| Train (opt) | `train/train_struct_refiner.py` | Train structural refiner / reranker | `artifacts/<run_id>/checkpoints/refiner/refiner_best.pth` |
| Eval | `eval/eval_full_entity_filtered.py` | Full-entity filtered evaluation (truth source) | Printed metrics + optional logs |
| Eval | `eval/eval_topk_inject.py` | Final system eval (topK-only injection + gate + Δ + CI) | Printed metrics + paired bootstrap CI |

Mainline default reports AVG (RHS+LHS)/2 with RHS/LHS splits, Hits@{1,3,10}, Recall@K, and paired bootstrap ΔMRR CI.

## Evaluation Protocol
对照组定义：
- R0/A：结构-only（RotatE；必要时 Refiner-only 作为补充对照）
- C：Sem+Gate（Δ 关闭）
- D：Sem+Gate+Δ（保持 sem-preserving threshold）

系统口径（TopK injection exact-rank）：
- RotatE 全实体打分取 topK；Sem/Gate/Δ 只在 topK 内。
- full-entity rank 阈值：`s_thresh = struct_w * s_gold_struct + (b * g) * s_gold_sem`（Δ 不参与阈值）。
- 若启用 `--gold_struct_threshold_no_sem`：`s_thresh` 退化为 `struct_w * s_gold_struct`。
- 评测脚本：`eval/eval_topk_inject.py`（仍是 full-entity filtered rank）。
- 严格等价验收：`eval/eval_topk_inject.py --strict_r0` 与 `eval/eval_full_entity_filtered.py` 的 ranks/AVG 必须一致（脚本：`scripts/check_eval_equivalence.sh`）。

统计输出：
- RHS/LHS/AVG + Hits@{1,3,10} + Recall@K
- paired bootstrap over queries for ΔMRR（D−C）

### Reproducibility Mode (Recommended for Paper Numbers)
评测脚本支持可选复现开关，不传参数时与当前默认口径一致：
- `--deterministic --disable_tf32 --matmul_precision highest --seed 42`
- 统一使用 `--batch_size 8 --score_eps 0`（full-entity 用 `--test_batch_size 8`）
- 等价性验收脚本：`scripts/check_eval_equivalence.sh`

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
bash scripts/run_mainline_full.sh fb15k_custom fb15k_custom_main_seed42
```

### Python One-Click (Optional)
```bash
python scripts/run_mainline.py --dataset fb15k_custom
```

### Output artifacts
- RotatE: `artifacts/<run_id>/checkpoints/rotate/best_model.pth`
- Bi-encoder: `artifacts/<run_id>/checkpoints/sem_biencoder/biencoder_best.pth`
- Gate: `artifacts/<run_id>/checkpoints/gate/gate_best.pth`
- Refiner: `artifacts/<run_id>/checkpoints/refiner/refiner_best.pth`
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
- RotatE defaults: `emb_dim=1000`, `margin=9.0`, `batch_size=1024`, `num_neg=256`, `lr=1e-4`, `epochs=200`, `adv_sampling=true`, `adv_temp=1.0`

## System-Protocol Comparison (topK injection exact-rank, run_id=200, test)
| Setting | AVG MRR | H@1 | H@3 | H@10 | Rec@200 |
|---|---|---|---|---|---|
| RotatE@TopK (strict) | 0.3075 | 0.2030 | 0.3521 | 0.5126 | 0.8363 |
| Sem-only strong@TopK | 0.2190 | 0.1445 | 0.2283 | 0.3607 | 0.8363 |
| C: Sem+Gate | 0.3204 | 0.2245 | 0.3564 | 0.5136 | 0.8363 |
| D: Sem+Gate+Δ | 0.3599 | 0.2728 | 0.3922 | 0.5367 | 0.8363 |
| Safety: entropy q=0.6 | 0.3361 | 0.2408 | 0.3733 | 0.5282 | 0.8363 |
Interpretation: D should surpass the strongest single-model baseline under the same system protocol (best-of RotatE@TopK or Sem-only@TopK), and be compared directly against C.

### Paired Bootstrap (system protocol)
- C − RotatE@TopK: 0.0129 (95% CI [0.0121, 0.0138])
- D − RotatE@TopK: 0.0525 (95% CI [0.0504, 0.0547])
- C − Sem-only strong@TopK: 0.1014 (95% CI [0.0981, 0.1047])
- D − C: 0.0396 (95% CI [0.0375, 0.0416])

## Sem-only Ablations
定义两条口径，表格中务必区分命名：
- **Sem-only (Full-entity)**：纯语义全实体评测，用于“语义单体上限”消融。评测脚本：`eval/eval_sem_biencoder_full.py`
- **Sem (TopK rerank)**：候选由 RotatE topK 给出，语义仅在 topK 内重排；不等价于“语义单体”。
- 主线 Sem ckpt 选优以 **topK proxy 的 AVG** 为准；Sem-only 仅作消融证据，不参与主线选优。
Full-entity 的 Sem-only 结果单独放在 Appendix。

### Sem-only Results (topK rerank, run_id=200, test)
| Setting | Protocol | AVG MRR | H@1 | H@3 | H@10 | Rec@200 |
|---|---|---|---|---|---|---|
| Sem-only (independent, topK-only rerank, b=0.5) | topK rerank | 0.1872 | 0.1228 | 0.1883 | 0.3069 | 0.8363 |
| Sem-only (independent, strong, topK-only rerank, b=0.5) | topK rerank | 0.2190 | 0.1445 | 0.2283 | 0.3607 | 0.8363 |
Note: b sweep on valid (0.5/1/2/4/8) tied; choose b=0.5 by lowest-b tie-break.

**Interpretation (system-aligned):**
- Sem-only@TopK compares **only within RotatE topK candidates**. This is a deliberate constraint to match the system protocol.
- This does **not** weaken the main conclusion because C/D are evaluated on the **same candidate set** (topK injection exact-rank), so the comparison is fair and directly reflects the final reported protocol.

### Appendix: Sem-only Full-entity (run_id=200, test)
| Setting | Protocol | AVG MRR | H@1 | H@3 | H@10 | Rec@200 |
|---|---|---|---|---|---|---|
| Sem-only (independent, full-entity) | full-entity | 0.1475 | 0.0817 | 0.1646 | 0.2614 | 0.6789 |
| Sem-only (independent, strong) | full-entity | 0.1800 | 0.1003 | 0.2053 | 0.3249 | 0.7385 |