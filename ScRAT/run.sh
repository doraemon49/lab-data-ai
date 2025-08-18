# 실행 bash run.sh

# HMIL 논문의 ScRAT AUC score 결과 ; COVID : 0.83+-0.04
# HMIL 논문의 ScRAT AUC score 결과 ; Cardio : 0.85+-0.02

# HMIL 논문의 Baseline Model ScRAT Hyperparameter
# – Learning rate: {1e-4, 1e-3, 1e-2}
# – Number of epochs: {100}
# – Number of attention heads: {1, 2, 4}
# – Dropout rate: {0.0, 0.3, 0.5, 0.7}
# – Weight decay: {1e-4, 1e-3, 1e-2}
# – Whether data augmentaion is performed: {True, False}
# – Embedding dimension: {8, 32, 64}
# – Number of augmented samples: {100}
# – PCA: {False}

##### split data 수정해야할 것
# dataloader.py
# -- 라벨 정의, cell type 정의
# main.py
# -- data 경로 정의

########################## Cardio split Dataset #############################

# log_file 이름 수정하기
# max_split_size_mb:128
# --batch_size 1
# 그냥 head를 1로함
# same_pheno -1 이건 안 함

log_file="logs_cardio_split/manual_annotation.txt"
echo "▶ Running with: lr=1e-4, heads=4, dropout=0.0, weight_decay=1e-4, emb_dim=8"
echo "▶ Output: ${log_file}"
CUDA_VISIBLE_DEVICES=0,1,2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
python main.py \
  --task custom_cardio \
  --learning_rate 1e-4 \
  --epochs 100 \
  --heads 1 \
  --dropout 0.0 \
  --weight_decay 1e-4 \
  --emb_dim 8 \
  --pca False \
  --repeat 5 \
  --cell_type_annotation manual_annotation \
  --batch_size 1 > "$log_file" 2>&1


# log_file="logs_cardio_split/manual_annotation_augment.txt"
# echo "▶ Running with: lr=1e-4, heads=4, dropout=0.0, weight_decay=1e-4, emb_dim=8"
# echo "▶ Output: ${log_file}"
# CUDA_VISIBLE_DEVICES=0,1,2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
# python main.py \
#   --task custom_cardio \
#   --learning_rate 1e-4 \
#   --epochs 100 \
#   --heads 4 \
#   --dropout 0.0 \
#   --weight_decay 1e-4 \
#   --emb_dim 8 \
#   --pca False \
#   --repeat 5 \
#   --cell_type_annotation manual_annotation \
#   --augment_num 100 \
#   --batch_size 1 \
#   --same_pheno -1 > "$log_file" 2>&1



########################## Covid split Dataset #############################


# log_file="logs_covid_split/singler_annotation_augment.txt"
# echo "▶ Running with: lr=1e-4, heads=4, dropout=0.0, weight_decay=1e-4, emb_dim=8"
# echo "▶ Output: ${log_file}"
# CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
# python main_split_covid.py \
#   --task custom_covid \
#   --learning_rate 1e-4 \
#   --epochs 100 \
#   --heads 4 \
#   --dropout 0.0 \
#   --weight_decay 1e-4 \
#   --emb_dim 8 \
#   --pca False \
#   --repeat 5 \
#   --cell_type_annotation singler_annotation \
#   --augment_num 100 > "$log_file" 2>&1

########################## COVID Dataset #############################


#     learning_rate  head  dropout  weight_decay  emb_dim  test_auc
# 80         0.0001     4      0.0        0.0001        8  0.944444
# 94         0.0001     4      0.5        0.0010       64  0.933333
# 57         0.0001     2      0.5        0.0010       32  0.933333


########################## Cardio Dataset #############################

# lr 1e-4, 1, dropout 0.3, 1e-3, 8,  augmentation 안 함.
# CUDA_VISIBLE_DEVICES=0,1,2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python main.py \
#   --dataset /data/project/kim89/cardio.h5ad \
#   --task custom_cardio \
#   --learning_rate 1e-4 \
#   --epochs 100 \
#   --heads 1 \
#   --dropout 0.3 \
#   --weight_decay 1e-3 \
#   --emb_dim 8 \
#   --pca False