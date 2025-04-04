import pandas as pd

# # 원본 메타데이터 로드
df = pd.read_csv("data/covid/20210701_NasalSwab_MetaData.txt", sep='\t')

# 타입 행과 실제 데이터 분리
column_types = df.iloc[0]       # 0번째 행 → 컬럼 타입
df_data = df.iloc[1:].copy()    # 1번째 행부터가 진짜 데이터
df_data.reset_index(drop=True, inplace=True)

print(f" 🔍 예제 데이터: {df_data.head(3).to_dict()}")


# 각 컬럼을 알맞은 타입으로 변환 (numeric 컬럼은 float로 변환)
for col in column_types[column_types == "numeric"].index:
    df_data[col] = pd.to_numeric(df_data[col], errors="coerce")  # 숫자로 변환, 안 되면 NaN

print("전체 세포 수:", len(df_data),"\n")  
print(f"유일한 환자(donor) 수: {df_data["donor_id"].nunique()}\n")
print("유일한 biosample 수:", df_data["biosample_id"].nunique(), "\n")

# 라벨 분포 확인
print(f"🦠 COVID 감염 여부 분포: {df_data["SARSCoV2_PCR_Status"].value_counts()}\n")
print(f"질병코드 : {df_data["disease"].value_counts()} \n")
print(f"질병이름 : {df_data["disease__ontology_label"].value_counts()} \n")

# 18개
print(f"🧪 (Coarse_Cell_Annotations) 종류:{len(df_data["Coarse_Cell_Annotations"].unique())}개. \n{df_data['Coarse_Cell_Annotations'].value_counts()}")


missing = df_data.isnull().sum()
print("📉 누락값이 있는 컬럼:")
print(missing[missing > 0])

print(f"📋 전체 컬럼명 리스트: {len(df_data.columns)} 개")
for col in df_data.columns:
    print(col)

import pandas as pd

# 원본 메타데이터 로드
# df = pd.read_csv("data/covid/20210220_NasalSwab_RawCounts.txt", sep='\t')
df = pd.read_csv("data/covid/20210220_NasalSwab_RawCounts.txt", sep='\t', index_col=0)
print(df.shape)     # (genes × cells) : (32871, 32588)

#  column = cell, 안쪽 딕셔너리의 key = gene
# print(f" 🔍 예제 데이터: {df.head(3).to_dict()}")   # 'GCCCCTTGTGAT_COVID19_Participant30': {'A1BG': 0.0, 'A1BG-AS1': 0.0, 'A1CF': 0.0},
print(f"📋 전체 ROW 리스트 (gene): {len(df.index)} 개")       # 📋 전체 ROW 리스트 (gene): 32871 개
print(f"📋 전체 컬럼명 리스트 (cell): {len(df.columns)} 개")  # CELL 바코드 32588

# print(df.index)
# print(df.columns)


import scanpy as sc
adata = sc.read_h5ad("covid_notnan.h5ad")
print(adata.X.shape)           # (n_cells, n_genes)     # (26944, 77)
print(adata.var_names)        # gene names (columns)
print(adata.obs_names)        # cell barcodes (rows)






import pandas as pd

# 원본 메타데이터 로드
df = pd.read_csv("data/covid/20210220_NasalSwab_NormCounts.txt", sep='\t')
print(df.shape)


# print(f" 🔍 예제 데이터: {df.head(3).to_dict()}")
print(f"📋 전체 ROW 리스트 (gene): {len(df.index)} 개")       # GENE 49
print(f"📋 전체 컬럼명 리스트 (cell): {len(df.columns)} 개")  # CELL 바코드 32588

print(df.index)
print(df.columns)