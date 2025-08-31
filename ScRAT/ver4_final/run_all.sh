log_file="logs_covid_split/manual_annotation.txt"
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main_optuna.py \
  --task custom_covid \
  --dataset /data/project/kim89/0804_data \
  --cell_type_annotation manual_annotation > "$log_file" 2>&1                              14718MiB |

log_file="logs_covid_split/singler_annotation.txt"
CUDA_VISIBLE_DEVICES=0,1,2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main_optuna.py \
  --task custom_covid \
  --dataset /data/project/kim89/0804_data \
  --cell_type_annotation singler_annotation > "$log_file" 2>&1                                14772MiB |

log_file="logs_kidney_split/manual_annotation.txt"
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main_optuna.py \
  --task custom_kidney \
  --dataset /data/project/kim89/0819_kidney \
  --cell_type_annotation manual_annotation > "$log_file" 2>&1

log_file="logs_kidney_split/singler_annotation.txt"
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main_optuna.py \
  --task custom_kidney \
  --dataset /data/project/kim89/0819_kidney \
  --cell_type_annotation singler_annotation > "$log_file" 2>&1
 
log_file="logs_cardio_split/manual_annotation.txt"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python main_optuna.py \
  --task custom_cardio \
  --dataset /data/project/kim89/0805_data \
  --cell_type_annotation manual_annotation \
  --batch_size 1 > "$log_file" 2>&1

log_file="logs_cardio_split/singler_annotation.txt"
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python main_optuna.py \
  --task custom_cardio \
  --dataset /data/project/kim89/0805_data \
  --cell_type_annotation singler_annotation \
  --batch_size 1 > "$log_file" 2>&1