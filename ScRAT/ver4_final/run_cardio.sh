log_file="logs_cardio_split/manual_annotation.txt"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python main_optuna.py \
  --task custom_cardio \
  --dataset /data/project/kim89/0805_data \
  --cell_type_annotation manual_annotation \
  --batch_size 1 > "$log_file" 2>&1

#  23329MiB / 24564MiB 
# 하나의 fold에서 80분(=20분*(3개parameter + 1test)
# 전체 (25fold)에서는, 2000분(=33시간)



log_file="logs_cardio_split/singler_annotation.txt"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python main_optuna.py \
  --task custom_cardio \
  --dataset /data/project/kim89/0805_data \
  --cell_type_annotation singler_annotation \
  --batch_size 1 > "$log_file" 2>&1