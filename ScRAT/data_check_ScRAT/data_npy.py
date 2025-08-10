import numpy as np

# 데이터 로드
combat_pca = np.load("data/COMBAT/COMBAT_X_pca.npy")
haniffa_pca = np.load("data/Haniffa/Haniffa_X_pca.npy")

# 데이터 형태 확인
print(f"COMBAT 데이터 크기: {combat_pca.shape}")  # (세포 개수, PCA 차원)
print(f"Haniffa 데이터 크기: {haniffa_pca.shape}")  # (세포 개수, PCA 차원)
