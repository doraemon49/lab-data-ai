# pip install scanpy
# python data_check_cardio.py
# *데이터 경로 수정해주셔야 합니다.

### pca 추가 코드
# import scanpy as sc
# # 데이터 로드
# adata = sc.read_h5ad('/data/project/kim89/cardio.h5ad')
# # PCA 실행
# sc.pp.pca(adata, n_comps=50)  # 필요에 따라 n_comps 조절
# # 결과 저장
# adata.write('/data/project/kim89/cardio_pca.h5ad')

### 데이터 확인 코드
# # obs: 셀 메타데이터
# print("✅ obs columns:")
# print(adata.obs.columns.tolist())
# print()

# # var: 유전자 메타데이터
# print("✅ var columns:")
# print(adata.var.columns.tolist())
# print()

# # obsm: 저차원 임베딩 (PCA, UMAP 등)
# print("✅ obsm keys:")
# print(list(adata.obsm.keys()))
# print()

# # layers: 추가 데이터 레이어
# print("✅ layers keys:")
# print(list(adata.layers.keys()))
# print()

# # raw: raw 데이터 여부
# print("✅ raw exists:")
# print(adata.raw is not None)
# if adata.raw is not None:
#     print("raw shape:", adata.raw.shape)
# print()

# # X matrix info
# print("✅ X matrix type:", type(adata.X))
# print("✅ X shape:", adata.X.shape)
"""
✅ obs columns:
['biosample_id', 'patient', 'cell_type', 'cell_type_annotation', 'sex', 'cell_type_leiden06', 'sub_cluster', 'n_umi', 'n_genes', 'cellranger_percent_mito', 'exon_prop', 'entropy', 'doublet_score', 'species', 'species__ontology_label', 'disease', 'disease__ontology_label', 'organ', 'organ__ontology_label', 'library_preparation_protocol', 'library_preparation_protocol__ontology_label', 'label']

✅ var columns:
['n_cells']

✅ obsm keys:
['X_pca']

✅ layers keys:
[]

✅ raw exists:
False

✅ X matrix type: <class 'scipy.sparse._csc.csc_matrix'>
✅ X shape: (592689, 32151)
"""


# # 데이터 축소 코드
# 6.2G
import scanpy as sc
# 1. 원본 데이터 로드
adata = sc.read_h5ad('/data/project/kim89/cardio.h5ad')
# 2. obs에서 필요한 컬럼만 유지
keep_obs = ['patient', 'disease__ontology_label', 'cell_type_annotation']
adata.obs = adata.obs[keep_obs]
# 3. var 정보 제거 (필요 시)
adata.var = adata.var[[]]
# 4. PCA 실행 (X 그대로 사용)
sc.pp.pca(adata, n_comps=50)
# 5. 저장
adata.write('/data/project/kim89/cardio_minimal.h5ad')


# 확인
import scanpy as sc
import numpy as np
import pandas as pd
adata = sc.read_h5ad('/data/project/kim89/cardio_minimal.h5ad')
print(adata.obs['disease__ontology_label'].value_counts())
# 환자 ID, 레이블 정보 추출
patient_ids = adata.obs['patient']
labels = adata.obs['disease__ontology_label']

# 셀 수 카운트
patient_counts = patient_ids.value_counts()

# 500개 미만 환자 필터링
under_500 = patient_counts[patient_counts < 500]

print("500개 미만 셀을 가진 환자 수:", len(under_500))
print(under_500)

# 라벨 별 분포 확인
adata.obs['label_mapped'] = labels.map({
    'normal': 0,
    'hypertrophic cardiomyopathy': 1,
    'dilated cardiomyopathy': 2
})
df = pd.DataFrame({'patient': patient_ids, 'label': adata.obs['label_mapped']})
patient_label = df.groupby('patient')['label'].first()

print("label 분포 (500개 이상 셀 보유한 환자 기준):")
print(patient_label[~patient_label.index.isin(under_500.index)].value_counts())
"""
disease__ontology_label
hypertrophic cardiomyopathy    235252
normal                         185441
dilated cardiomyopathy         171996
Name: count, dtype: int64
500개 미만 셀을 가진 환자 수: 0
Series([], Name: count, dtype: int64)
/data/project/kim89/ScRAT/data_check.py:115: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  patient_label = df.groupby('patient')['label'].first()
label 분포 (500개 이상 셀 보유한 환자 기준):
label
0    16
1    15
2    11
Name: count, dtype: int64
"""











"""
Cardio 데이터 파일에는 최소 아래 컬럼이 필요합니다:

obs['patient_id']: 환자 ID

obs['Outcome']: 예측할 label (예: 0, 1 / control, disease)

obs['cell_type']: cell type 정보 (mixup 용)

👉 만약 컬럼 이름이 다르다면 h5ad를 로드해서 컬럼 이름 확인:

patient_id	    -> patient
label (Outcome) ->	disease__ontology_label
cell_type	    -> cell_type_annotation

"""
# # 컬럼 값 확인
# print(adata.obs.columns)
# """
# Index(['biosample_id', 'patient', 'cell_type', 'cell_type_annotation', 'sex',
#        'cell_type_leiden06', 'sub_cluster', 'n_umi', 'n_genes',
#        'cellranger_percent_mito', 'exon_prop', 'entropy', 'doublet_score',
#        'species', 'species__ontology_label', 'disease',
#        'disease__ontology_label', 'organ', 'organ__ontology_label',
#        'library_preparation_protocol',
#        'library_preparation_protocol__ontology_label', 'label'],
#       dtype='object')
# """

# # 각 컬럼 값 확인
# print("🔹 biosample_id unique values:")
# print(adata.obs['biosample_id'].unique())
# """
# 🔹 biosample_id unique values:
# ['LV_1622_2_nf', 'LV_1422_1_hcm', 'LV_1722_2_hcm', 'LV_1462_1_hcm', 'LV_1558_2_nf', ..., 'LV_1472_1_dcm', 'LV_1735_2_hcm', 'LV_1600_2_nf', 'LV_1606_1_dcm', 'LV_1561_2_nf']
# Length: 80
# Categories (80, object): ['LV_1290_1_dcm', 'LV_1290_2_dcm', 'LV_1300_1_dcm', 'LV_1300_2_dcm', ...,
#                           'LV_1726_1_hcm', 'LV_1726_2_hcm', 'LV_1735_1_hcm', 'LV_1735_2_hcm']
# """

# print("\n🔹 patient unique values:")
# print(adata.obs['patient'].unique())
# """
# 🔹 patient unique values:
# ['P1622', 'P1422', 'P1722', 'P1462', 'P1558', ..., 'P1539', 'P1726', 'P1504', 'P1472', 'P1606']
# Length: 42
# Categories (42, object): ['P1290', 'P1300', 'P1304', 'P1358', ..., 'P1718', 'P1722', 'P1726', 'P1735']
# """

# print("\n🔹 disease unique values:")
# print(adata.obs['disease'].unique())
# """
# 🔹 disease unique values:
# ['PATO_0000461', 'MONDO_0005045', 'MONDO_0005021']
# Categories (3, object): ['MONDO_0005021', 'MONDO_0005045', 'PATO_0000461']
# """

# print("\n🔹 disease__ontology_label unique values:")
# print(adata.obs['disease__ontology_label'].unique())
# """
# 🔹 disease__ontology_label unique values:
# ['normal', 'hypertrophic cardiomyopathy', 'dilated cardiomyopathy']
# Categories (3, object): ['dilated cardiomyopathy', 'hypertrophic cardiomyopathy', 'normal']
# """

# print("\n🔹 label unique values:")
# print(adata.obs['label'].unique())
# """
# 🔹 label unique values:
# [0 1 2]
# """

# print("\n🔹 cell_type unique values:")
# print(adata.obs['cell_type'].unique())
# """
# 🔹 cell_type unique values:
# ['CL_0000746', 'CL_0000136', 'CL_0000235', 'CL_0002350', 'CL_2000066', ..., 'CL_0000359', 'CL_0000669', 'CL_0000097', 'CL_0000542', 'CL_0000077']
# Length: 13
# Categories (13, object): ['CL_0000077', 'CL_0000097', 'CL_0000136', 'CL_0000235', ..., 'CL_0002350',
#                           'CL_0010008', 'CL_0010022', 'CL_2000066']
# """

# print("\n🔹 cell_type_annotation unique values:")
# print(adata.obs['cell_type_annotation'].unique())
# """
# 🔹 cell_type_annotation unique values:
# ['cardiac muscle cell', 'fat cell', 'macrophage', 'endocardial cell', 'cardiac ventricle fibroblast', ..., 'vascular associated smooth muscle cell', 'pericyte cell', 'mast cell', 'lymphocyte', 'mesothelial cell']
# Length: 13
# Categories (13, object): ['cardiac endothelial cell', 'cardiac muscle cell', 'cardiac neuron',
#                           'cardiac ventricle fibroblast', ..., 'mast cell', 'mesothelial cell', 'pericyte cell',
#                           'vascular associated smooth muscle cell']
# """
