# Download Rshinydata_singlecell-20231219T155916Z-001.zip from https://zenodo.org/records/10407126 and unzip the folder

import pandas as pd
import os
import scanpy as sc

dfs = []
for f in os.listdir("Rshinydata_singlecell"):
    df = pd.read_csv("Rshinydata_singlecell/"+f)
    dfs.append(df)

df = pd.concat(dfs, axis=0)
df = df.iloc[:, 1:]
df.set_index("Row.names", inplace=True)

# meta는 cell 단위의 메타데이터
meta = df.iloc[:,:195]
# X는 cell × gene 유전자 발현 행렬
X = df.iloc[:,195:]

# 치료 전(pre) 데이터만 사용
meta = meta[meta["sample_id_pre_post"].apply(lambda x: x.split("_")[-1] == "Pre")]
# Favourable/Unfavourable 반응에 따라 이진 레이블(label) 부여
meta["label"] = meta["Combined_outcome"].apply(lambda x: 0 if x=="Unfavourable" else 1 if x=="Favourable" else -1)
meta = meta[meta["label"] != -1]

# 샘플 ID를 환자 ID로 매핑
sample_to_patient = pd.read_csv("icb_sample_id_to_patient_id.csv").set_index("sample_id")
meta["patient"] = meta["sample_id_pre_post"].apply(lambda x: sample_to_patient.loc[x, "patient"])

# 세포 타입 주석 추가
# ct = pd.read_csv("singler_icb_pre.csv", index_col=0)
ct = pd.read_csv("singler_icb.csv", index_col=0)	# 작성
meta["cell_type_annotation"] = ct["pruned.labels"]  # meta의 index와 ct의 index가 모두 cell ID이기 때문에, → 자동으로 index 기준으로 정렬 후 병합됩니다.
meta = meta[meta["cell_type_annotation"].notna()]

# 유전자 필터링 (유전자 목록에 있는 것만 가져옴)
genes = pd.read_csv("genes_icb.csv")["Gene"].to_list()
# AnnData 객체 생성
adata = sc.AnnData(X.loc[meta.index, genes], obs=meta)
adata.var.index = genes

# label만 명시적으로 정수 처리!
adata.obs["label"] = adata.obs["label"].astype(int)

# object 타입 컬럼은 전부 문자열로 변환 (write_h5ad 에러 방지)
for col in adata.obs.select_dtypes(include=["object"]).columns:
    adata.obs[col] = adata.obs[col].astype(str)

adata.write_h5ad("../icb.h5ad")

# ✅ 확인
print(adata.obs.columns)
print(adata.obs[["patient", "label", "cell_type_annotation"]].head())

