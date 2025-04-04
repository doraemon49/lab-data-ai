# python data/covid_inf/h5ad_analysis.py
import scanpy as sc

adata = sc.read_h5ad("data/covid_inf/challenge_pbmc_cellxgene_230223.h5ad")

# from scipy import sparse
# import numpy as np

# # sparse matrix인 경우
# if sparse.issparse(adata.X):
#     print("✅ X is sparse matrix!")
#     X_dense = adata.X.toarray()
# else:
#     print("✅ X is dense array!")
#     X_dense = adata.X

# # NaN, Inf 검사
# print("NaN 개수:", np.isnan(X_dense).sum())
# print("Inf 개수:", np.isinf(X_dense).sum())



print(adata)                
"""
AnnData object with n_obs × n_vars = 371892 × 33696
    obs: 'patient_id', 'time_point', 'covid_status', 'sex', 'cell_state_wVDJ', 'cell_state', 'cell_state_woIFN', 'cell_type', 'cell_compartment', 'sequencing_library', 'Institute', 'ObjectCreateDate'
    var: 'name'
    obsm: 'X_umap_harmony_rna_wvdj_30pcs_6000hvgs'
"""

print(adata.X.shape)        # (371892, 33696)
# import pandas as pd
# df_X = pd.DataFrame(adata.X, columns=adata.var.index)
# print(df_X.head())

print("==== column : 33696개 유전자 * 이름 ====")
print(adata.var.shape)      # (33696, 1)     → 유전자 정보 (유전자 이름과 각 몇개)
# print(adata.var.index)      # 유전자 이름      # Index(['A1BG', 'A1BG-AS1', 'A1CF', 'A2M', 'A2M-AS1', ...], dtype='object')
print(adata.var.head())     # 각 유전자별 ?
"""
                    name
MIR1302-2HG  MIR1302-2HG
FAM138A          FAM138A
OR4F5              OR4F5
AL627309.1    AL627309.1
AL627309.3    AL627309.3
"""

print("==== row : 371892개 세포 * 정보 12가지 ====")
print(adata.obs.shape)          # (371892, 12)  → 각 세포의 메타데이터
# print(adata.obs.columns)        # 메타데이터 컬럼
"""
Index(['patient_id', 'time_point', 'covid_status', 'sex', 'cell_state_wVDJ',
       'cell_state', 'cell_state_woIFN', 'cell_type', 'cell_compartment',
       'sequencing_library', 'Institute', 'ObjectCreateDate'],
      dtype='object')
"""
# print(adata.obs.head())         

print(adata.obs["patient_id"].nunique())                # 16명

print(adata.obs["cell_type"].nunique())                 # 27개
# cell_types = adata.obs["cell_type"].unique()
# print(cell_types)
# print(adata.obs["cell_type"].value_counts())

print(adata.obs["covid_status"].nunique())              # 3개
print(adata.obs["covid_status"].unique())   # ['Sustained infection', 'Transient infection', 'Abortive infection']

# label_counts = adata.obs["covid_status"].value_counts()
# print(label_counts)
"""
covid_status
Abortive infection     155563
Sustained infection    135919
Transient infection     80410
Name: count, dtype: int64
"""

print(adata.obs["sequencing_library"].nunique())        # 40
print(adata.obs["sequencing_library"].value_counts())

print(adata.obs["Institute"].value_counts())        # 세포 전체 다 같은 값
"""
Institute
Wellcome Sanger Institute    371892
Name: count, dtype: int64
"""


# 각 patient 별 label 분포 보기
# patient_label_counts = adata.obs.groupby(["patient_id", "covid_status"]).size().unstack(fill_value=0)
# print(patient_label_counts)


# 각 patient 별 label 분포 보기
# patient_label_counts = adata.obs.groupby(["patient", "disease__ontology_label"]).size().unstack(fill_value=0)
# print(patient_label_counts)



# # 각 patient 별 label 분포 보기
# patient_label_counts = adata.obs.groupby(["patient", "label"]).size().unstack(fill_value=0)
# print(patient_label_counts)

# # 각 cell 별 label 분포 보기
# cell_label_counts = adata.obs.groupby(["cell", "label"]).size().unstack(fill_value=0)
# print(cell_label_counts)




# print(adata.obs["disease__ontology_label"].value_counts())
# # disease__ontology_label
# # COVID-19    18072
# # normal       8872
# # Name: count, dtype: int64
