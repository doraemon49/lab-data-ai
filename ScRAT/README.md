# 1. main_original.py

ScRAT 깃허브의 원본 main.py

# 2. main_customized.py

원본에서, 단일 데이터 (~.h5ad)를 넣어주는 것으로 수정.
즉, 데이터를 넣어주는 경로 및 파라미터 정도만 간단하게 수정됨.

# 3. main_split_covid.py

split 된 데이터 (cross validation의 효과를 보기 위해 5\*5=25개의 데이터)를 위한 코드.

단, covid data (이진분류)에 대해서만 수정함.

# 4. main_split_data.py

split 된 데이터를 위한 코드.

이진분류뿐만 아니라, multi-class(다중 분류)도 가능해짐.

# 5. main_optuna.py

split 데이터를 각 fold마다 optuna를 사용하여 최적의 hyperparameter를 찾아나감.
(각 fold마다 최적 hyperparameter 바탕으로 test를 찍음. 총 25개의 test 결과.)
