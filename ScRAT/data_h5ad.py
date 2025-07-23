import tarfile
import os
import scanpy as sc
import pandas as pd

# # 압축 해제
# tar_path = 'data/SC4/COVID19_ALL.h5ad.tar.gz'
# extract_dir = 'data/SC4'

# print(f"📦 압축 해제 중: {tar_path}")
# with tarfile.open(tar_path, 'r:gz') as tar:
#     tar.extractall(path=extract_dir)
# print("✅ 압축 해제 완료")

# .h5ad 파일 찾기
extract_dir = 'data/Haniffa'
h5ad_files = [f for f in os.listdir(extract_dir) if f.endswith('.h5ad')]
if not h5ad_files:
    print("❌ .h5ad 파일을 찾을 수 없습니다.")
else:
    h5ad_path = os.path.join(extract_dir, h5ad_files[0])
    print(f"📂 분석할 파일: {h5ad_path}")

    # 데이터 로드
    adata = sc.read_h5ad(h5ad_path, backed='r')

    # 전체 셀 수
    # print(f"\n🧬 전체 cell 수: {adata.n_obs}")
    # print(f"🧪 전체 feature 수 (유전자 등): {adata.n_vars}")
    # print(f"📝 obs 컬럼 목록: {list(adata.obs_keys())}")


    # 환자 ID 정보가 있는지 확인 # Hanaffa : sample_id (130명), patient_id(130명) # SC4 : PatientID
    if 'sample_id' in adata.obs.columns:
        patient_counts = adata.obs['sample_id'].value_counts()
        print(f"\n👤 환자 수: {adata.obs['sample_id'].nunique()}")
        print(f"👥 환자별 셀 수:\n{patient_counts}")

    # Haniffa 
    # print(adata.obs['Status'].value_counts())
    # print(adata.obs['Outcome'].value_counts())
    # print(adata.obs['sample_id'].value_counts()) # sample_id와 patient_id 비슷한듯 하나 # Length: 143, 130으로 다름.
    # print(adata.obs['patient_id'].value_counts()) # Length : 130


    # SC4
    
    # 주요 celltype 분포
    # print("📊 세포 타입 분포:")
    # print(adata.obs['celltype'].value_counts())
    # print("🔢 고유한 세포 타입 수:", adata.obs['celltype'].nunique())
    # print("🔹 고유 세포 타입 목록:")
    # print(adata.obs['celltype'].unique())

    # print(adata.obs['SARS-CoV-2'].value_counts()) # SARS-CoV-2 감염 여부
    # print(adata.obs['CoVID-19 severity'].value_counts()) # 중증도 수준 (예: "Healthy", "Mild", "Moderate", "Severe", "Critical", "ICU" 등)
    # print(adata.obs['datasets'].value_counts())
    # print(adata.obs['Outcome'].value_counts())
    # print(adata.obs['Sample type'].value_counts())
    # print(adata.obs['Sample time'].value_counts())



"""
Haniffa 데이터
 For COMBAT and Haniffa datasets, we perform the task of disease diagnosis
    COVID versus Non-COVID
---------------------

🧬 전체 cell 수: 647366
🧪 전체 feature 수 (유전자 등): 24929
📝 obs 컬럼 목록: ['sample_id', 'n_genes', 'n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt', 'full_clustering', 'initial_clustering', 'Resample', 'Collection_Day', 'Sex', 'Age_interval', 'Swab_result', 'Status', 'Smoker', 'Status_on_day_collection', 'Status_on_day_collection_summary', 'Days_from_onset', 'Site', 'time_after_LPS', 'Worst_Clinical_Status', 'Outcome', 'patient_id']

👤 환자 수: 143
👥 환자별 셀 수:
sample_id
MH9143427    14317
AP6          14086
MH8919333    12081
MH9143277    11710
AP11         10921
             ...  
MH8919233      307
MH8919229      305
MH8919232      184
MH8919228      147
MH8919277       65
Name: count, Length: 143, dtype: int64

👤 환자 수: 130
👥 환자별 셀 수:
patient_id
MH9143427    14317
AP6          14086
MH8919333    12081
MH9143277    11710
AP11         10921
             ...  
MH8919233      307
MH8919229      305
MH8919232      184
MH8919228      147
MH8919277       65
Name: count, Length: 130, dtype: int64

Status                                                                                                           
Covid        527286                                                                                              
Healthy       97039                                                                                              
Non_covid     15157                                                                                              
LPS            7884                                                                                              
Name: count, dtype: int64    

Outcome                                                                                                          
Home       504847
unknown    100683
Death       41836
Name: count, dtype: int64

sample_id                                               
MH9143427    14317
AP6          14086
MH8919333    12081
MH9143277    11710
AP11         10921
             ...  
MH8919233      307
MH8919229      305                                      
MH8919232      184
MH8919228      147
MH8919277       65
Name: count, Length: 143, dtype: int64

patient_id                                              
MH9143427    14317                                      
AP6          14086
MH8919333    12081
MH9143277    11710
AP11         10921
             ...  
MH8919233      307
MH8919229      305                                      
MH8919232      184                                      
MH8919228      147
MH8919277       65
Name: count, Length: 130, dtype: int64
"""


"""
SC4 데이터
For SC4 which includes mostly COVID samples
    mild/moderate versus severe/critical (경증/중등증 vs. 중증/위중증)
    convalescence versus progression (회복 vs. 진행)
------------------

🧬 전체 cell 수: 1462702
🧪 전체 feature 수 (유전자 등): 27943
📝 obs 컬럼 목록: ['celltype', 'majorType', 'sampleID', 'PatientID', 'datasets', 'City', 'Age', 'Sex', 'Sample type', 'CoVID-19 severity', 'Sample time', 'Sampling day (Days after symptom onset)', 'SARS-CoV-2', 'Single cell sequencing platform', 'BCR single cell sequencing', 'TCR single cell sequencing', 'Outcome', 'Comorbidities', 'COVID-19-related medication and anti-microbials', 'Leukocytes [G/L]', 'Neutrophils [G/L]', 'Lymphocytes [G/L]', 'Unpublished']
👤 환자 수: 196
👥 환자별 셀 수:
PatientID
P-M004    49223
P-M007    31763
P-M010    31359
P-S022    29408
P-S086    28625
          ...  
P-S003      447
P-S006      403
P-M057      356
P-S002      252
P-S009      168
Name: count, Length: 196, dtype: int64

📊 세포 타입 분포:
celltype
B_c01-TCL1A                    227948
Mono_c3-CD14-VCAN              136158
T_CD4_c01-LEF1                 107008
B_c02-MS4A1-CD27                92913
Mono_c2-CD14-HLA-DPB1           84402
                                ...  
T_CD4_c14-MKI67-CCL5_h            191
DC_c3-LAMP3                       186
Neu_c5-GSTP1(high)OASL(low)        59
Epi-AT2                            25
Mast                               17
Name: count, Length: 64, dtype: int64

 고유한 세포 타입 수: 64
🔹 고유 세포 타입 목록:
['Mono_c1-CD14-CCL3', 'B_c02-MS4A1-CD27', 'B_c01-TCL1A', 'Mono_c2-CD14-HLA-DPB1', 'Macro_c2-CCL3L1', ..., 'Epi-Secretory', 'Macro_c6-VCAN', 'Neu_c6-FGF23', 'Epi-AT2', 'Mast']
Length: 64
Categories (64, object): ['B_c01-TCL1A', 'B_c02-MS4A1-CD27', 'B_c03-CD27-AIM2',
                          'B_c04-SOX5-TNFRSF1B', ..., 'T_CD8_c11-MKI67-FOS', 'T_CD8_c12-MKI67-TYROBP',
                          'T_CD8_c13-HAVCR2', 'T_gdT_c14-TRDV2']

SARS-CoV-2   
positive    1297702
negative     165000
Name: count, dtype: int64

CoVID-19 severity
mild/moderate      700968
severe/critical    596734
control            165000
Name: count, dtype: int64

datasets
d10    277384
d13    189374
d07    156007
d17    154011
d03    123422
d02    112351
d15     92204
d11     79916
d08     74822
d04     68463
d09     48168
d06     31286
d14     25538
d01     14767
d05     12347
d12      2642
Name: count, dtype: int64

Outcome
discharged    1216871
control        165000
deceased        80831
Name: count, dtype: int64

Sample type                                                                                
fresh PBMC                                                    542075
frozen PBMC                                                   451096
B cells sorted from frozen PBMC (MACS, STEMCELL 19054)        196908
CD3+ T cell and CD19+ B cell sorted from fresh PBMC (FACS)     74822
CD19+ B cell sorted from fresh PBMC (MACS)                     65822
fresh BALF                                                     42723
CD3+ T cell sorted from fresh PBMC (FACS)                      36550
CD19+ B cell sorted from fresh PBMC (FACS)                     31913
fresh Sputum                                                   14502
fresh PFMC                                                      6291
Name: count, dtype: int64

Sample time
convalescence    787987
progression      509715
control          165000
Name: count, dtype: int64
"""