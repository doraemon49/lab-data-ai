log_file="logs_covid_split/manual_annotation.txt"
CUDA_VISIBLE_DEVICES=1,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main_optuna.py \
  --task custom_covid \
  --dataset /data/project/kim89/0804_data \
  --cell_type_annotation manual_annotation > "$log_file" 2>&1

# 하나의 fold에서 20분(=5분*3개parameter + test5분) # 25개의 fold 500분(=8.3시간) # 실제로 7시간.
# |    0   N/A  N/A     13106      C   python                              
# |    1   N/A  N/A     13106      C   python                               
# |    2   N/A  N/A     13106      C   python                              
# |    3   N/A  N/A     13106      C   python                                

log_file="logs_covid_split/singler_annotation.txt"
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
python main_optuna.py \
  --task custom_covid \
  --dataset /data/project/kim89/0804_data \
  --cell_type_annotation singler_annotation > "$log_file" 2>&1

# tuning 단계
# |    1   N/A  N/A    650909      C   python                                       5704MiB |
# |    3   N/A  N/A    650909      C   python                                    5148MiB |
# 본 학습 단계
# |    1   N/A  N/A    650909      C   python                                    18032MiB |  
# |    3   N/A  N/A    650909      C   python                                     17450MiB |