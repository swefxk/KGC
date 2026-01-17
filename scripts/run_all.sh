#!/usr/bin/env bash
set -euo pipefail

# Adjust these paths as needed
DATA_PATH="data/fb15k_custom"
ROTATE_CKPT="checkpoints/fair_experiment/rotate_stage3_std_v2/best_model.pth"
REFINER_CKPT="checkpoints/fair_experiment/refiner_stage3_std_k16_filteredneg_seed42/refiner_best.pth"

# Stage A cache (train+valid filtered)
python build/build_rotate_cache_rhs_topk.py \
  --data_path "$DATA_PATH" \
  --pretrained_rotate "$ROTATE_CKPT" \
  --K 500 --use_filtered --filtered_splits train_valid \
  --out_path "$DATA_PATH/cache/train_rhs_top500_neg_filtered_trainvalid.pt"

# Stage A hard-only bi-encoder
python train/train_sem_biencoder.py \
  --data_path "$DATA_PATH" \
  --pretrained_rotate "$ROTATE_CKPT" \
  --train_cache "$DATA_PATH/cache/train_rhs_top500_neg_filtered_trainvalid.pt" \
  --save_dir checkpoints/sem_biencoder_stageA_hardonly_ep10 \
  --hard_k 128 --queue_size 0 \
  --tau 0.1 --proj_dim 256 --text_norm \
  --batch_size 256 --lr 3e-4 \
  --epochs 10 --eval_every 1 --eval_sem_rhs_only

BIENCODER_CKPT="checkpoints/sem_biencoder_stageA_hardonly_ep10/biencoder_best.pth"

# Valid b-scale search (topK=500)
for b in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0; do
python eval/eval_topk_inject.py \
    --data_path "$DATA_PATH" --eval_split valid \
    --pretrained_rotate "$ROTATE_CKPT" \
    --pretrained_sem "$BIENCODER_CKPT" \
    --b_scale "$b" --topk 500 --batch_size 16
done

# Test verification (b*=2.0)
python eval/eval_topk_inject.py \
  --data_path "$DATA_PATH" --eval_split test \
  --pretrained_rotate "$ROTATE_CKPT" \
  --pretrained_sem "$BIENCODER_CKPT" \
  --b_scale 2.0 --topk 500 --batch_size 16

# Refiner-only vs Refiner+Sem (b=2.0)
python eval/eval_topk_inject.py \
  --data_path "$DATA_PATH" --eval_split test \
  --pretrained_rotate "$ROTATE_CKPT" \
  --pretrained_refiner "$REFINER_CKPT" \
  --pretrained_sem "$BIENCODER_CKPT" \
  --b_scale 0.0 --topk 500 --batch_size 16

python eval/eval_topk_inject.py \
  --data_path "$DATA_PATH" --eval_split test \
  --pretrained_rotate "$ROTATE_CKPT" \
  --pretrained_refiner "$REFINER_CKPT" \
  --pretrained_sem "$BIENCODER_CKPT" \
  --b_scale 2.0 --topk 500 --batch_size 16
