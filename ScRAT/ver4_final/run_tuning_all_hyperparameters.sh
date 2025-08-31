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

# (run_hyperparameter_tuning.sh) 결과 : 
#7번째로 가장 test_auc가 높긴 하나, 가장 메모리 효율적이라 선택. (cardio에서 실행됨)
#     learning_rate  head  dropout  weight_decay  emb_dim  test_auc
# 33         0.0001     1      0.7        0.0001       32  0.911111



########################## Cardio split Dataset #############################

# max_split_size_mb:128 에서 32로 수정함
# --batch_size 1로 설정함

########################## kidney split Dataset #############################

# |    2   N/A  N/A   2642904      C   python                                    22756MiB |
# |    3   N/A  N/A   2642904      C   python                                    19998MiB |

# < mixup() 실행 시 >
# |    2   N/A  N/A   2676606      C   python                                    22150MiB |
# |    3   N/A  N/A   2676606      C   python                                    20216MiB |


########################## Covid split Dataset #############################

# |    0   N/A  N/A   2657033      C   python                                    23154MiB |
# |    1   N/A  N/A   2657033      C   python                                    23542MiB |


# < mixup() 실행 시 : 메모리 차지가 더 작다...(?) >
# |    0   N/A  N/A   2666702      C   python                                    12722MiB |
# |    1   N/A  N/A   2666702      C   python                                    10510MiB |



########################## COVID Dataset (hyperparameter tunning) #############################

# run_hyperparameter_tuning.sh 결과 top 10

#     learning_rate  head  dropout  weight_decay  emb_dim  test_auc
# 80         0.0001     4      0.0        0.0001        8  0.944444
# 94         0.0001     4      0.5        0.0010       64  0.933333
# 57         0.0001     2      0.5        0.0010       32  0.933333
# 86         0.0001     4      0.3        0.0010        8  0.927778
# 91         0.0001     4      0.5        0.0100       64  0.922222
# 77         0.0001     4      0.0        0.0010        8  0.916667
# < 7순위. 메모리 사용량이 효율적(=적어서=cardio데이터 돌아감)이라서 선택 >
# 33         0.0001     1      0.7        0.0001       32  0.911111
# 79         0.0001     4      0.0        0.0001       64  0.911111
# 76         0.0001     4      0.0        0.0010       64  0.911111
# 37         0.0001     2      0.0        0.0100       64  0.911111

lr=0.0001
head=1
dropout=0.7
wd=0.0001
emb=32

log_file="test/lr${lr}_head${head}_drop${dropout}_wd${wd}_emb${emb}_all_patients_repeat25.txt"

echo "▶ Running with: lr=${lr}, heads=${head}, dropout=${dropout}, weight_decay=${wd}, emb_dim=${emb}"
echo "▶ Output: ${log_file}"

# 실행
CUDA_VISIBLE_DEVICES=1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main_customized_covid.py \
--dataset /data/project/kim89/ScRAT/data/covid.h5ad \
--task custom_covid \
--learning_rate $lr \
--epochs 100 \
--heads $head \
--dropout $dropout \
--weight_decay $wd \
--emb_dim $emb \
--pca False \
--repeat 5 > "$log_file" 2>&1

########################## Cardio Dataset (안 돌림) #############################

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