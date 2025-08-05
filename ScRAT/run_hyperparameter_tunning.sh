#!/bin/bash

# COVID 데이터에 대해, 하이퍼파라미터별 결과 도출.

# 사용할 하이퍼파라미터 목록
learning_rates=(1e-4 1e-3 1e-2)
heads=(1 2 4)
dropouts=(0.0 0.3 0.5 0.7)
weight_decays=(1e-4 1e-3 1e-2)
emb_dims=(8 32 64)
# 일단 data augmentation 제외함.


# 로그 디렉토리 생성
mkdir -p logs

# 반복문 시작
for lr in "${learning_rates[@]}"; do
  for head in "${heads[@]}"; do
    for dropout in "${dropouts[@]}"; do
      for wd in "${weight_decays[@]}"; do
        for emb in "${emb_dims[@]}"; do

          # 파일 이름 생성
          log_file="logs/lr${lr}_head${head}_drop${dropout}_wd${wd}_emb${emb}.txt"

          echo "▶ Running with: lr=${lr}, heads=${head}, dropout=${dropout}, weight_decay=${wd}, emb_dim=${emb}"
          echo "▶ Output: ${log_file}"

          # 실행
          CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
          python main.py \
            --dataset /data/project/kim89/ScRAT/data/covid.h5ad \
            --task custom_covid \
            --learning_rate $lr \
            --epochs 100 \
            --heads $head \
            --dropout $dropout \
            --weight_decay $wd \
            --emb_dim $emb \
            --pca False > "$log_file" 2>&1

        done
      done
    done
  done
done

# > 리다이렉션이 **stdout(표준 출력)**을 터미널 대신 파일로 보내어,
# logs/ 폴더에 각 조합별 결과를 저장합니다.
# 파일 이름은 하이퍼파라미터 값으로 구성돼 예: lr1e-3_head2_drop0.3_wd1e-2_emb64.txt.
# 2>&1은 에러도 로그에 포함시킵니다.