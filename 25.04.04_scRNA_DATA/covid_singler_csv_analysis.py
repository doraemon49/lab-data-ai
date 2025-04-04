import pandas as pd

# CSV 파일 로드
file_path = "data/covid/singler_covid.csv"
df = pd.read_csv(file_path)


# 상위 5개 데이터 출력
print("\n📌 상위 5개 데이터 샘플:")
print(df.head())


# 전체 세포 수 출력
print("\n📌총 세포 수:", df.shape[0])
# 고유한 세포 수 출력
unique_cells = df["Unnamed: 0"].nunique()
print("\n📌 고유한 세포 수:", unique_cells)


# 고유한 label 개수 확인
unique_labels = df["labels"].nunique()
print("\n📌고유한 labels 개수:", unique_labels)
# labels 컬럼의 고유값 출력
unique_labels = df["labels"].unique()
print("\n📌 labels 컬럼의 36개 고유값:")
print(unique_labels)
"""
면역 세포: 'T_cells', 'B_cell', 'NK_cell', 'Macrophage', 'Neutrophils', 'Monocyte', 'DC'
조혈 세포: 'HSC_-G-CSF', 'HSC_CD34+', 'CMP', 'MEP', 'GMP', 'Myelocyte', 'Pro-Myelocyte'
줄기 세포: 'Embryonic_stem_cells', 'Tissue_stem_cells', 'MSC', 'iPS_cells'
기타 세포: 'Epithelial_cells', 'Endothelial_cells', 'Fibroblasts', 'Astrocyte', 'Neurons', 'Platelets'
"""

# 컬럼 이름 출력
print("📌 CSV 파일 컬럼 목록:")
print(df.columns.tolist())
