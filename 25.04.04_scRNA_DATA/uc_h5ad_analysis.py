# python data/uc/uc_h5ad_analysis.py
import scanpy as sc
import pandas as pd
adata = sc.read_h5ad("uc_fib.h5ad")

print(adata)                
"""
<<epi>>
AnnData object with n_obs × n_vars = 123006 × 20028
    obs: 'Cluster', 'nGene', 'nUMI', 'Subject', 'Health', 'Location', 'Sample'

<<fib>>
AnnData object with n_obs × n_vars = 31872 × 19076
    obs: 'Cluster', 'nGene', 'nUMI', 'Subject', 'Health', 'Location', 'Sample'

<<imm>>
AnnData object with n_obs × n_vars = 210614 × 20529
    obs: 'Cluster', 'nGene', 'nUMI', 'Subject', 'Health', 'Location', 'Sample'

"""

print(adata.X.shape)        
"""
(123006, 20028)
(31872, 19076)
(210614, 20529)
"""
# import pandas as pd
# df_X = pd.DataFrame(adata.X, columns=adata.var.index)
# print(df_X.head())

print("==== column : 20028, 19076, 20529 개 유전자 ====")
print(adata.var.shape)          # (20028, 0) # (19076, 0) # (20529, 0)
# print(adata.var.index)        # 유전자     
print(adata.var.head())     
"""
Empty DataFrame
Columns: []
Index: [7SK, A1BG, A1BG-AS1, A1CF, A2M]
"""

print("==== row : 123006, 31872, 210614 개 세포 * 정보 7 가지 ====")
print(adata.obs.shape)          # (123006, 7) # (31872, 7) # (210614, 7) → 각 세포의 메타데이터
print(adata.obs.columns)        # 메타데이터 컬럼
"""
Index(['Cluster', 'nGene', 'nUMI', 'Subject', 'Health', 'Location', 'Sample'], dtype='object')
"""
# print(adata.obs.head())         
                                        # <<epi>>    # <<fib>>   # <<imm>>
print(adata.obs["Cluster"].nunique())   # 15        # 13        # 23
print(adata.obs["nGene"].nunique())     # 6108      # 3368      # 3355
print(adata.obs["nUMI"].nunique())      # 25407     # 8365      # 18064
print(adata.obs["Subject"].nunique())   # 30        # 30        # 30
print(adata.obs["Location"].nunique())  # 2         # 2         # 2
print(adata.obs["Sample"].nunique())    # 131       # 77        # 132

print(adata.obs["Health"].nunique())    # 3가지     # 3가지      # 3가지
print(adata.obs["Health"].unique())   
# ['Non-inflamed', 'Inflamed', 'Healthy']
# Categories (3, object): ['Healthy', 'Inflamed', 'Non-inflamed']

label_counts = adata.obs["Health"].value_counts()
print(label_counts)
"""
<<epi>>
Health
Healthy         50258
Non-inflamed    49704
Inflamed        23044
Name: count, dtype: int64

<<fib>>
Health
Non-inflamed    13147
Inflamed        10245
Healthy          8480
Name: count, dtype: int64

<<imm>>
Health
Inflamed        91830
Non-inflamed    67412
Healthy         51372
Name: count, dtype: int64
"""

# 각 patient 별 label 분포 보기
patient_label_counts = adata.obs.groupby(["Subject", "Health"]).size().unstack(fill_value=0)
print(patient_label_counts)

"""
Health   Healthy  Inflamed  Non-inflamed
Subject                                 
N7             0       260           536
N8           959         0             0
N9             0       966          4804
N10         6624         0             0
N11         5187         0             0
N12            0       356           423
N13         1483         0             0
N14            0       978          1852
N15         6082         0             0
N16         3094         0             0
N17         2323         0             0
N18         3416         0             0
N19            0      1693          1881
N20         2740         0             0
N21         4250         0             0
N23            0      1722          4656
N24            0      3472          4381
N26            0      3256          3515
N44            0      1309          1282
N46         3044         0             0
N49            0       831           171
N50            0       707          1419
N51        11056         0             0
N52            0       380         10620
N58            0      1685          5621
N106           0      1261           482
N110           0      1622          5303
N111           0      1945           586
N539           0       601           788
N661           0         0          1384

Health   Healthy  Inflamed  Non-inflamed
Subject                                 
N7             0       282           770
N8           503         0             0
N9             0       121           600
N10         2195         0             0
N11          348         0             0
N12            0       556           333
N13          274         0             0
N14            0       399           182
N15          735         0             0
N16          363         0             0
N17          357         0             0
N18          922         0             0
N19            0       213           214
N20          580         0             0
N21          316         0             0
N23            0        29           299
N24            0       317          1124
N26            0       171           542
N44            0       623           433
N46         1198         0             0
N49            0       331          1106
N50            0       407           319
N51          689         0             0
N52            0       685          2643
N58            0      2196           375
N106           0       109           488
N110           0       371           236
N111           0      1345          1085
N539           0       872           871
N661           0      1218          1527

Health   Healthy  Inflamed  Non-inflamed
Subject                                 
N7             0      1291          1307
N8           762         0             0
N9             0      3845          1504
N10         7824         0             0
N11         1264         0             0
N12            0       443           253
N13         2938         0             0
N14            0       899           642
N15         3832         0             0
N16         1960         0             0
N17         3101         0             0
N18         3721         0             0
N19            0       517          2862
N20         6243         0             0
N21         2386         0             0
N23            0      4590          1884
N24            0      1722          2349
N26            0       933          2807
N44            0      4460          6330
N46         6075         0             0
N49            0      2853          3492
N50            0      3803          3846
N51        11266         0             0
N52            0     11836          7166
N58            0     13718          1671
N106           0      3478          1724
N110           0      1841          1031
N111           0     16358          4067
N539           0      1872          2790
N661           0     17371         21687

"""

# scMILD 논문에서 제시된 내용 확인하기
# Sample 수 : 42                                        # 안 맞음    # 30 = 18+12
# Classes : Inflamed 18, Healthy 24                     # Inflamed 는 맞는데, Healthy가 안 맞음. # ✅ 순수 Inflamed 환자 수: 18 # ✅ 순수 Healthy 환자 수: 12
# Cells : 18,725                                        # 맞음      # 전체 세포 수: 18725
# Avg Cells : 446 (Average number of cell per sample)   # 안 맏음   # 평균 세포 수: 624.17

import scanpy as sc
import pandas as pd


# "Healthy", "Inflamed"만 필터링
adata = adata[adata.obs["Health"].isin(["Healthy", "Inflamed"])]

# 환자별로 Health 상태에 따른 cell 수 세기
patient_health_counts = adata.obs.groupby(["Subject", "Health"]).size().unstack(fill_value=0)
# 환자별 라벨 종류 개수
label_type_count_per_patient = (patient_health_counts > 0).sum(axis=1)
# 각 환자 상태 판별
patient_labels = label_type_count_per_patient.map({1: "Pure", 2: "Mixed"})
# Pure 환자 중 어떤 라벨인지 확인
pure_patients = patient_health_counts[label_type_count_per_patient == 1]
pure_patient_types = pure_patients.idxmax(axis=1)  # 각 row에서 가장 큰 값의 column 이름 (Health or Inflamed)
# 최종 요약
summary = pure_patient_types.value_counts().to_dict()
mixed_count = (patient_labels == "Mixed").sum()
# 출력
print(f"✅ 순수 Healthy 환자 수: {summary.get('Healthy', 0)}")      # ✅ 순수 Healthy 환자 수: 12
print(f"✅ 순수 Inflamed 환자 수: {summary.get('Inflamed', 0)}")    # ✅ 순수 Inflamed 환자 수: 18
print(f"⚠️  혼합(Mixed) 환자 수: {mixed_count}")                    # ⚠️  혼합(Mixed) 환자 수: 0

# 전체 샘플 수
n_samples = adata.obs["Subject"].nunique()
# 전체 세포 수
n_cells = adata.n_obs
# 평균 세포 수 (전체 / 샘플 수) # average number of cells  per sample
avg_cells_per_sample = n_cells / n_samples
# 출력
print(f"전체 세포 수: {n_cells}")                                   # 전체 세포 수: 18725
print(f"평균 세포 수: {avg_cells_per_sample:.2f}")                  # 평균 세포 수: 624.17

# ===================
# ===================
# uc의 Metadata
# ===================
# ===================

df = pd.read_csv("data/uc/all.meta2.txt", sep='\t')
# 타입 행과 실제 데이터 분리
column_types = df.iloc[0]       # 0번째 행 → 컬럼 타입
df_data = df.iloc[1:].copy()    # 1번째 행부터가 진짜 데이터
df_data.reset_index(drop=True, inplace=True)
print(f" 🔍 예제 데이터: {df_data.head(3).to_dict()}")
"""
 🔍 예제 데이터: {'NAME': {0: 'N7.EpiA.AAACATACACACTG', 1: 'N7.EpiA.AAACCGTGCATCAG', 2: 'N7.EpiA.AAACGCACAATCGC'}, 'Cluster': {0: 'TA 1', 1: 'TA 1', 2: 'TA 2'}, 'nGene': {0: '328', 1: '257', 2: '300'}, 'nUMI': {0: '891', 1: '663', 2: '639'}, 'Subject': {0: 'N7', 1: 'N7', 2: 'N7'}, 'Health': {0: 'Non-inflamed', 1: 'Non-inflamed', 2: 'Non-inflamed'}, 'Location': {0: 'Epi', 1: 'Epi', 2: 'Epi'}, 'Sample': {0: 'N7.EpiA', 1: 'N7.EpiA', 2: 'N7.EpiA'}}
"""
print("전체 세포 수:", len(df_data),"\n")                           # 전체 세포 수: 365493 
print(f"유일한 환자(Subject) 수: {df_data["Subject"].nunique()}\n")   # 유일한 환자(Sample) 수: 30
# print("유일한 biosample 수:", df_data["biosample_id"].nunique(), "\n")