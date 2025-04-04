import scanpy as sc

adata = sc.read_h5ad("cardio.h5ad")

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





print(adata)                # AnnData object with n_obs × n_vars = 25718 × 75
print(adata.X.shape)        # (25718, 75)  

print("adata.X.shape:", adata.X.shape)
print("adata.var.index:", adata.var.index.tolist())
print("len(var.index):", len(adata.var.index))


import pandas as pd
df_X = pd.DataFrame(adata.X.toarray(), columns=adata.var.index.to_list())
print(df_X.head())

print("==== column : 75개 유전자 * 갯수(n_cells) ====")
print(adata.var.shape)      # (75, 1)     → 유전자 정보 (유전자 이름과 각 몇개)
print(adata.var.index)      # 유전자 이름     
print(adata.var.head())     # 각 유전자별 갯수
"""
            n_cells
AL627309.1      121
AL627309.5      192
AP006222.2       25
AC114498.1        6
AL669831.2       10
"""

print("==== row : 25718개 세포 * 정보 22가지 ====")
print(adata.obs.shape)          # (25718, 22)  → 각 세포의 메타데이터
print(adata.obs.columns)        # 메타데이터 컬럼
print(adata.obs.head())         # 행 샘플
"""

                      biosample_id patient  ... library_preparation_protocol__ontology_label label
TTCTTCCGTTCAACGT-1-0  LV_1622_2_nf   P1622  ...                                    10x 3' v3     0
CATCCACCATCTAACG-1-0  LV_1622_2_nf   P1622  ...                                    10x 3' v3     0
"""

# print(adata.obs["patient"].nunique())              

print(adata.obs["cell_type_annotation"].nunique())  # 13개
cell_types = adata.obs["cell_type_annotation"].unique()
print(cell_types)
print(adata.obs["cell_type_annotation"].value_counts())

print(adata.obs["label"].nunique())                 # 2개
label_counts = adata.obs["label"].value_counts()
print(label_counts)
"""
label
1    18508
0     7210
"""




# # 각 patient 별 label 분포 보기
# patient_label_counts = adata.obs.groupby(["patient", "label"]).size().unstack(fill_value=0)
# print(patient_label_counts)

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
