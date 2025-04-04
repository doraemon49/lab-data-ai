import scanpy as sc

adata = sc.read_h5ad("icb.h5ad")
print(adata)                # 전체 구조 요약
print(adata.X.shape)        # (9292, 824)
import pandas as pd
df_X = pd.DataFrame(adata.X, columns=adata.var.index)
print(df_X.head())

print("==== column : 824개 유전자 ====")
print(adata.var.shape)      # (824, 0)     → 유전자 정보 (보통 이름만 있으면 (824, 0))
print(adata.var.index)  # 유전자 이름 5개만 보기      # Index(['HAVCR2', 'CTLA4', 'PDCD1', 'IDO1', 'CXCL10'], dtype='object')
print(adata.var.head())     # 유전자 이름

print("==== row : 9292개 세포의 정보 197가지 ====")
print(adata.obs.shape)      # (9292, 197)  → 각 세포의 메타데이터
print(adata.obs.columns[:50])    # 메타데이터 컬럼
print(adata.obs.columns[50:100])    # 메타데이터 컬럼
print(adata.obs.columns[100:150])    # 메타데이터 컬럼
print(adata.obs.columns[150:])    # 메타데이터 컬럼
print(adata.obs.head())     # 행 샘플

print(adata.obs["patient"].nunique())               # 57명
print(adata.obs["cell_type_annotation"].nunique())  # 23개
print(adata.obs["label"].nunique())                 # 2개
