# pip install scanpy
# python data_check_covid.py
# *데이터 경로 수정해주셔야 합니다.

# import scanpy as sc
# # 데이터 로드
# adata = sc.read_h5ad('/data/project/kim89/covid.h5ad')
# # PCA 실행
# sc.pp.pca(adata, n_comps=50)  # n_comps는 ScRAT에서 필요하면 조절
# # 결과 저장
# adata.write('/data/project/kim89/covid_pca.h5ad')


# 축소 전 ;확인
import scanpy as sc
import numpy as np
import pandas as pd
adata = sc.read_h5ad('/data/project/kim89/ScRAT/data/covid_pca.h5ad')
print(adata.obs['disease__ontology_label'].value_counts())
# 환자 ID, 레이블 정보 추출
patient_ids = adata.obs['donor_id']
labels = adata.obs['disease__ontology_label']

# 셀 수 카운트
patient_counts = patient_ids.value_counts()

# 500개 미만 환자 필터링
under_500 = patient_counts[patient_counts < 500]

print("500개 미만 셀을 가진 환자 수:", len(under_500))
# print(under_500)

# 라벨 별 분포 확인
adata.obs['label_mapped'] = labels.map({
    'normal': 0,
    'COVID-19': 1
})
df = pd.DataFrame({'patient': patient_ids, 'label': adata.obs['label_mapped']})
patient_label = df.groupby('patient')['label'].first()

print("label 분포 (500개 이상 셀 보유한 환자 기준):")
print(patient_label[~patient_label.index.isin(under_500.index)].value_counts())
"""
< 요약 >
전체 환자 수	    50 = 19+31
500셀 이상 환자 수	19명 (= 12명 + 7명)

클래스 분포	
label=1 (COVID-19): 12명
label=0 (normal): 7명

=> 300셀 이상 가진 환자 수는 32(=추가된 13 + 원래 19)
"""
"""
disease__ontology_label
COVID-19    18073
normal       8874
Name: count, dtype: int64
500개 미만 셀을 가진 환자 수: 31
donor_id
COVID19_Participant25    496
COVID19_Participant8     462
COVID19_Participant39    454
Control_Participant4     440
Control_Participant1     414
COVID19_Participant31    410
COVID19_Participant7     409
Control_Participant8     393
COVID19_Participant11    382
COVID19_Participant14    382
COVID19_Participant40    381
COVID19_Participant2     329
COVID19_Participant23    307
Control_Participant14    274
COVID19_Participant18    227
COVID19_Participant36    200
Control_Participant6     131
COVID19_Participant30    116
COVID19_Participant3     104
COVID19_Participant12    102
Control_Participant12     99
COVID19_Participant27     93
COVID19_Participant5      89
Control_Participant13     88
COVID19_Participant9      86
COVID19_Participant6      84
COVID19_Participant35     83
COVID19_Participant38     51
Control_Participant2      41
COVID19_Participant21     39
COVID19_Participant10     18
Name: count, dtype: int64
/data/project/kim89/ScRAT/data_check_covid.py:63: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
  patient_label = df.groupby('patient')['label'].first()
label 분포 (500개 이상 셀 보유한 환자 기준):
label
1    12
0     7
Name: count, dtype: int64

"""

# # 데이터 축소 코드
# 5.8G 그대로...
import scanpy as sc

# 1. 원본 데이터 로드
adata = sc.read_h5ad('/data/project/kim89/covid.h5ad')

# 2. obs에서 필요한 컬럼만 유지
keep_obs = ['donor_id', 'disease__ontology_label', 'cell_type_annotation']
adata.obs = adata.obs[keep_obs]

# 3. var 정보 제거 (필요 시)
adata.var = adata.var[[]]

# 4. PCA 실행 (X 그대로 사용)
sc.pp.pca(adata, n_comps=50)

# 5. 저장
adata.write('/data/project/kim89/covid_minimal.h5ad')



# 축소 후 ;확인
import scanpy as sc
import numpy as np
import pandas as pd
adata = sc.read_h5ad('/data/project/kim89/covid_minimal.h5ad')
print(adata.obs['disease__ontology_label'].value_counts())
# # 환자 ID, 레이블 정보 추출
patient_ids = adata.obs['donor_id']
labels = adata.obs['disease__ontology_label']

# 셀 수 카운트
patient_counts = patient_ids.value_counts()

# 500개 미만 환자 필터링
under_500 = patient_counts[patient_counts < 500]

print("500개 미만 셀을 가진 환자 수:", len(under_500))
# print(under_500)

# # 라벨 별 분포 확인
adata.obs['label_mapped'] = labels.map({
    'normal': 0,
    'COVID-19': 1,
})
df = pd.DataFrame({'patient': patient_ids, 'label': adata.obs['label_mapped']})
patient_label = df.groupby('patient')['label'].first()

print("label 분포 (500개 이상 셀 보유한 환자 기준):")
print(patient_label[~patient_label.index.isin(under_500.index)].value_counts())

"""
disease__ontology_label                                                                                                                                                                     
COVID-19    18073                                                                                                                                                                           
normal       8874                                                                                                                                                                           
Name: count, dtype: int64                                                                                                                                                                   
500개 미만 셀을 가진 환자 수: 31
donor_id
COVID19_Participant25    496
COVID19_Participant8     462
COVID19_Participant39    454
Control_Participant4     440
Control_Participant1     414
COVID19_Participant31    410
COVID19_Participant7     409
Control_Participant8     393COVID19_Participant11    382COVID19_Participant14    382
COVID19_Participant40    381
COVID19_Participant2     329
COVID19_Participant23    307
Control_Participant14    274
COVID19_Participant18    227
COVID19_Participant36    200
Control_Participant6     131
COVID19_Participant30    116
COVID19_Participant3     104
COVID19_Participant12    102
Control_Participant12     99
COVID19_Participant27     93
COVID19_Participant5      89
Control_Participant13     88
COVID19_Participant9      86
COVID19_Participant6      84
COVID19_Participant35     83
COVID19_Participant38     51
Control_Participant2      41
COVID19_Participant21     39
COVID19_Participant10     18
Name: count, dtype: int64
/data/project/kim89/ScRAT/data_check_covid.py:159: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False
 to retain current behavior or observed=True to adopt the future default and silence this warning.
  patient_label = df.groupby('patient')['label'].first()
label 분포 (500개 이상 셀 보유한 환자 기준):
label
0.0    7
Name: count, dtype: int64
"""












"""
마찬가지로, Covid 데이터 파일에서도 컬럼 값 확인

patient_id	-> donor_id	
Outcome     -> disease__ontology_label {'normal': 0, 'COVID-19': 1}
cell_type	-> cell_type_annotation	

"""
# import scanpy as sc
# # 데이터 로드
# adata = sc.read_h5ad('/data/project/kim89/covid.h5ad')

# print(adata.obs.columns)

# # 후보 컬럼 유니크 값 출력
# # 환자 ID (patient_id)
# print("🔹 donor_id unique values:")
# print(adata.obs['donor_id'].unique())
# print()

# print("🔹 patient unique values:")
# print(adata.obs['patient'].unique())
# print()

# # Outcome (예측 label)
# print("🔹 SARSCoV2_PCR_Status unique values:")
# print(adata.obs['SARSCoV2_PCR_Status'].unique())
# print()

# print("🔹 Cohort_Disease_WHO_Score unique values:")
# print(adata.obs['Cohort_Disease_WHO_Score'].unique())
# print()

# print("🔹 SARSCoV2_PCR_Status_and_WHO_Score unique values:")
# print(adata.obs['SARSCoV2_PCR_Status_and_WHO_Score'].unique())
# print()

# print("🔹 Peak_Respiratory_Support_WHO_Score unique values:")
# print(adata.obs['Peak_Respiratory_Support_WHO_Score'].unique())
# print()

# print("🔹 disease__ontology_label unique values:")
# print(adata.obs['disease__ontology_label'].unique())
# print()

# #  cell_type
# print("🔹 cell_type_annotation unique values:")
# print(adata.obs['cell_type_annotation'].unique())
# print()

# print("🔹 Coarse_Cell_Annotations unique values:")
# print(adata.obs['Coarse_Cell_Annotations'].unique())
# print()

# print("🔹 Detailed_Cell_Annotations unique values:")
# print(adata.obs['Detailed_Cell_Annotations'].unique())
# print()

"""
🔹 donor_id unique values:
['Control_Participant7', 'COVID19_Participant13', 'COVID19_Participant31', 'Control_Participant12', 'COVID19_Participant5', ..., 'COVID19_Participant18', 'COVID19_Participant27', 'COVID19_Participant21', 'COVID19_Participant20', 'COVID19_Participant30']
Length: 50
Categories (50, object): ['COVID19_Participant2', 'COVID19_Participant3', 'COVID19_Participant4',
                          'COVID19_Participant5', ..., 'Control_Participant12', 'Control_Participant13',
                          'Control_Participant14', 'Control_Participant15']

🔹 patient unique values:
['Control_Participant7', 'COVID19_Participant13', 'COVID19_Participant31', 'Control_Participant12', 'COVID19_Participant5', ..., 'COVID19_Participant18', 'COVID19_Participant27', 'COVID19_Participant21', 'COVID19_Participant20', 'COVID19_Participant30']
Length: 50
Categories (50, object): ['COVID19_Participant2', 'COVID19_Participant3', 'COVID19_Participant4',
                          'COVID19_Participant5', ..., 'Control_Participant12', 'Control_Participant13',
                          'Control_Participant14', 'Control_Participant15']

🔹 SARSCoV2_PCR_Status unique values:
['neg', 'pos']
Categories (2, object): ['neg', 'pos']

🔹 Cohort_Disease_WHO_Score unique values:
['Control_WHO_0', 'COVID19_WHO_6-8', 'COVID19_WHO_1-5']
Categories (3, object): ['COVID19_WHO_1-5', 'COVID19_WHO_6-8', 'Control_WHO_0']

🔹 SARSCoV2_PCR_Status_and_WHO_Score unique values:
['neg_0', 'pos_8', 'pos_6', 'pos_5', 'pos_4', 'pos_3', 'pos_1', 'pos_7']
Categories (8, object): ['neg_0', 'pos_1', 'pos_3', 'pos_4', 'pos_5', 'pos_6', 'pos_7', 'pos_8']

🔹 Peak_Respiratory_Support_WHO_Score unique values:
['0', '8', '6', '5', '4', '3', '1', '7']
Categories (8, object): ['0', '1', '3', '4', '5', '6', '7', '8']

🔹 disease__ontology_label unique values:
['normal', 'COVID-19']
Categories (2, object): ['COVID-19', 'normal']

🔹 cell_type_annotation unique values:
['Developing Ciliated Cells', 'Ciliated Cells', 'Secretory Cells', 'Squamous Cells', 'Goblet Cells', ..., 'Developing Secretory and Goblet Cells', 'Plasmacytoid DCs', 'Enteroendocrine Cells', 'Erythroblasts', 'Mast Cells']
Length: 18
Categories (18, object): ['B Cells', 'Basal Cells', 'Ciliated Cells', 'Dendritic Cells', ...,
                          'Plasmacytoid DCs', 'Secretory Cells', 'Squamous Cells', 'T Cells']

🔹 Coarse_Cell_Annotations unique values:
['Developing Ciliated Cells', 'Ciliated Cells', 'Secretory Cells', 'Squamous Cells', 'Goblet Cells', ..., 'Developing Secretory and Goblet Cells', 'Plasmacytoid DCs', 'Enteroendocrine Cells', 'Erythroblasts', 'Mast Cells']
Length: 18
Categories (18, object): ['B Cells', 'Basal Cells', 'Ciliated Cells', 'Dendritic Cells', ...,
                          'Plasmacytoid DCs', 'Secretory Cells', 'Squamous Cells', 'T Cells']

🔹 Detailed_Cell_Annotations unique values:
['Developing Ciliated Cells', 'FOXJ1 high Ciliated Cells', 'BEST4 high Cilia high Ciliated Cells', 'Cilia high Ciliated Cells', 'SERPINB11 high Secretory Cells', ..., 'Enteroendocrine Cells', 'Interferon Responsive Cytotoxic CD8 T Cells', 'Interferon Responsive Secretory Cells', 'Erythroblasts', 'Mast Cells']
Length: 39
Categories (39, object): ['AZGP1 SCGB3A1 LTF high Goblet Cells', 'AZGP1 high Goblet Cells', 'B Cells',
                          'BEST4 high Cilia high Ciliated Cells', ..., 'SCGB1A1 high Goblet Cells',
                          'SERPINB11 high Secretory Cells', 'SPRR2D high Squamous Cells',
                          'VEGFA high Squamous Cells']
"""

