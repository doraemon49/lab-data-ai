log_file="logs_covid_split_cellReduce/manual_annotation.txt"
echo "▶ Running with: lr=0.0001, heads=1, dropout=0.7, weight_decay=0.0001, emb_dim=32"
echo "▶ Output: ${log_file}"
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main.py \
  --task custom_covid \
  --dataset /data/project/kim89/0804_data_cellReduce \
  --learning_rate 0.0001 \
  --epochs 100 \
  --heads 1 \
  --dropout 0.7 \
  --weight_decay 0.0001 \
  --emb_dim 32 \
  --pca False \
  --repeat 5 \
  --cell_type_annotation manual_annotation > "$log_file" 2>&1


# log_file="logs_covid_split_cellReduce/manual_annotation_augment.txt"
# echo "▶ Running with: lr=0.0001, heads=1, dropout=0.7, weight_decay=0.0001, emb_dim=32"
# echo "▶ Output: ${log_file}"
# CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
# python main.py \
#   --task custom_covid \
#   --dataset /data/project/kim89/0804_data_cellReduce \
#   --learning_rate 0.0001 \
#   --epochs 100 \
#   --heads 1 \
#   --dropout 0.7 \
#   --weight_decay 0.0001 \
#   --emb_dim 32 \
#   --pca False \
#   --repeat 5 \
#   --cell_type_annotation manual_annotation \
#   --augment_num 100 > "$log_file" 2>&1


log_file="logs_covid_split_cellReduce/singler_annotation.txt"
echo "▶ Running with: lr=0.0001, heads=1, dropout=0.7, weight_decay=0.0001, emb_dim=32"
echo "▶ Output: ${log_file}"
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main.py \
  --task custom_covid \
  --dataset /data/project/kim89/0804_data_cellReduce \
  --learning_rate 0.0001 \
  --epochs 100 \
  --heads 1 \
  --dropout 0.7 \
  --weight_decay 0.0001 \
  --emb_dim 32 \
  --pca False \
  --repeat 5 \
  --cell_type_annotation singler_annotation > "$log_file" 2>&1


# log_file="logs_covid_split_cellReduce/singler_annotation_augment.txt"
# echo "▶ Running with: lr=0.0001, heads=1, dropout=0.7, weight_decay=0.0001, emb_dim=32"
# echo "▶ Output: ${log_file}"
# CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
# python main.py \
#   --task custom_covid \
#   --dataset /data/project/kim89/0804_data_cellReduce \
#   --learning_rate 0.0001 \
#   --epochs 100 \
#   --heads 1 \
#   --dropout 0.7 \
#   --weight_decay 0.0001 \
#   --emb_dim 32 \
#   --pca False \
#   --repeat 5 \
#   --cell_type_annotation singler_annotation \
#   --augment_num 100 > "$log_file" 2>&1