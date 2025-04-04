# python data/covid/covid_h5ad_analysis.py
import scanpy as sc

adata = sc.read_h5ad("covid_notnan.h5ad")

from scipy import sparse
import numpy as np

# sparse matrix인 경우
if sparse.issparse(adata.X):
    print("✅ X is sparse matrix!")
    X_dense = adata.X.toarray()
else:
    print("✅ X is dense array!")
    X_dense = adata.X

# NaN, Inf 검사
print("NaN 개수:", np.isnan(X_dense).sum())
print("Inf 개수:", np.isinf(X_dense).sum())





# print(adata)                # AnnData object with n_obs × n_vars = 26944 × 77 # obs: 'donor_id',... # var: 'n_cells'
print(adata.X.shape)        # (26944, 77)
# import pandas as pd
# df_X = pd.DataFrame(adata.X, columns=adata.var.index)
# print(df_X.head())

# print("==== column : 77개 유전자 * 갯수(n_cells) ====")
print(adata.var.shape)      # (77, 1)     → 유전자 정보 (유전자 이름과 각 몇개)
# print(adata.var.index)      # 유전자 이름      # Index(['A1BG', 'A1BG-AS1', 'A1CF', 'A2M', 'A2M-AS1', ...], dtype='object')
# print(adata.var.head())     # 각 유전자별 갯수
# """
#           n_cells
# A1BG          299
# A1BG-AS1      119
# A1CF           43
# A2M           295
# A2M-AS1       113
# """

print("==== row : 26944개 세포 * 정보 29가지 ====")
print(adata.obs.shape)          # (26944, 29)  → 각 세포의 메타데이터
# print(adata.obs.columns)        # 메타데이터 컬럼 # Index(['donor_id', 'Peak_Respiratory_Support_WHO_Score', ... , 'label', 'cell_type_annotation', 'patient'],  dtype='object')
# print(adata.obs.head())         # 행 샘플
# """
#                                                donor_id  ...               patient
# GTCGGGGGGTGG_Control_Participant7  Control_Participant7  ...  Control_Participant7
# CAAATCAATTAT_Control_Participant7  Control_Participant7  ...  Control_Participant7
# """

print(adata.obs["patient"].nunique())               # 50명
# print(adata.obs["patient"].value_counts())

print(adata.obs["cell_type_annotation"].nunique())  # 36개
# cell_types = adata.obs["cell_type_annotation"].unique()
# print(cell_types)
# print(adata.obs["cell_type_annotation"].value_counts())

print(adata.obs["label"].nunique())                 # 2개
label_counts = adata.obs["label"].value_counts()
print(label_counts)
"""
label
1    18072
0     8872
Name: count, dtype: int64
"""

print(adata.obs["SARSCoV2_PCR_Status"].value_counts())
"""
SARSCoV2_PCR_Status
pos    18072
neg     8872
"""
# # 각 patient 별 label 분포 보기
# patient_label_counts = adata.obs.groupby(["patient", "SARSCoV2_PCR_Status"]).size().unstack(fill_value=0)
# print(patient_label_counts)

print(adata.obs["disease__ontology_label"].value_counts())
"""
disease__ontology_label
COVID-19    18072
normal       8872
"""
# # 각 patient 별 label 분포 보기
# patient_label_counts = adata.obs.groupby(["patient", "disease__ontology_label"]).size().unstack(fill_value=0)
# print(patient_label_counts)

# 각 patient 별 label 분포 보기
patient_label_counts = adata.obs.groupby(["patient", "label"]).size().unstack(fill_value=0)
print(patient_label_counts)

# # 각 cell 별 label 분포 보기
# cell_label_counts = adata.obs.groupby(["cell", "label"]).size().unstack(fill_value=0)
# print(cell_label_counts)

# print(adata.obs["SARSCoV2_PCR_Status"].value_counts())
# # SARSCoV2_PCR_Status
# # pos    18072
# # neg     8872
# # Name: count, dtype: int64

# print(adata.obs["SingleCell_SARSCoV2_RNA_Status"].value_counts())
# # SingleCell_SARSCoV2_RNA_Status
# # neg    24044
# # amb     2487
# # pos      413
# # Name: count, dtype: int64

# print(adata.obs["disease__ontology_label"].value_counts())
# # disease__ontology_label
# # COVID-19    18072
# # normal       8872
# # Name: count, dtype: int64
