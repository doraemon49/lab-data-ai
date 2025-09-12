# python results_covid_split/auc_calculate.py
import numpy as np

# 다섯 개 값 입력
values = [0.8920,
0.8800,
0.7567,
0.8256,
0.7292]

# 평균 계산
mean_val = np.mean(values)

# 표본 표준편차 계산 (n-1로 나눔)
std_val = np.std(values, ddof=1)

print(f"평균: {mean_val:.4f}")
print(f"표준편차: {std_val:.4f}")
print(f"{mean_val:.4f} ± {std_val:.4f}")
