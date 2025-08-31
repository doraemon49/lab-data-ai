# ProtoCell4P: An Explainable Prototype-based Neural Network for Patient Classification Using Single-cell RNA-seq

This repo contains the source code for our [manuscript](https://doi.org/10.1093/bioinformatics/btad493) to Bioinformatics.

## Setup
### lupus dataset
- Change directory to `./data/lupus`
- Follow the instruction of `./data/lupus/download.txt`
### cardio dataset
- Change directory to `./data/cardio`
- Follow the instruction of `./data/cardio/download.txt`
### covid dataset
- Change directory to `./data/covid`
- Follow the instruction of `./data/covid/download.txt`

## Usage
- Change directory to `./src`
### Run ProtoCell4P
- Run `sh run.sh`
### Run BaseModel
- Run `sh run_base.sh`
### Run Ablation Studies
- Run `sh run_ablation.sh`

## Citation
If you find our research useful, please consider citing:

```
@article{xiong2023protocell4p,
  title={ProtoCell4P: an explainable prototype-based neural network for patient classification using single-cell RNA-seq},
  author={Xiong, Guangzhi and Bekiranov, Stefan and Zhang, Aidong},
  journal={Bioinformatics},
  volume={39},
  number={8},
  pages={btad493},
  year={2023},
  publisher={Oxford University Press}
}
```

# cell type을 쓰면 (1) 프리트레인에서 프로토타입이 CT별로 정렬되고, (2) 본학습에서도 보조 분류(CT)가 함께 학습되어 더 안정화됩니다.
# load_ct=True과 cell_type_annotation을 넣어줘야 한다
# --cell_type_annotation manual_annotation 또는 --cell_type_annotation singler_annotation


# 1. Pretrain (Config.pretrain)
# 목적: 모델이 표현 학습(embedding) 과 프로토타입 초기화를 먼저 안정적으로 하도록 돕는 단계.
# 사용하는 손실:입력 데이터를 다시 복원하는 재구성 손실 (reconstruction loss), 셀 임베딩과 프로토타입 간 거리 제약(c2p_loss, p2c_loss, p2p_loss), (옵션) 셀 타입 분류 손실(ct_loss)
# 즉, label(y) 분류 정확도를 직접적으로 최적화하지 않고, latent space를 구조화하는 데 집중합니다.
# 결과적으로 프로토타입들이 데이터 공간에서 잘 퍼지도록 하고, 셀/유전자 표현이 안정적으로 학습됩니다.

# 2. 본학습 (Config.train)
# 목적: 실제로 라벨(y) 분류를 잘 하도록 모델을 최적화.
# 사용하는 손실: CrossEntropyLoss (질병/집단 예측 같은 최종 분류 목적) # 거기에다가 pretrain 때의 손실 일부를 추가로 섞어서 정규화 역할을 하도록 합니다.
# 즉, 분류 성능을 올리면서도 latent space 구조를 유지하도록 합니다