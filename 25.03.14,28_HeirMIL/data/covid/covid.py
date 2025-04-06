# 📦 필요 라이브러리
import pandas as pd
import scanpy as sc

# ✅ 1. 데이터 로드
df = pd.read_csv("20210220_NasalSwab_RawCounts.txt", sep='\t')

# ✅ 2. AnnData 객체 생성 (행: cell, 열: gene)
adata = sc.AnnData(df.T)
adata.obs.index = df.columns
adata.var.index = df.index

# ✅ 3. 필수 전처리
sc.pp.filter_genes(adata, min_cells=5)

sc.pp.normalize_total(adata, target_sum=1e4)

sc.pp.log1p(adata)

# ✅ 4. 메타데이터 로드 & 연결
meta = pd.read_csv("20210701_NasalSwab_MetaData.txt", sep="\t").drop(index=0).reset_index(drop=True)
meta.set_index("NAME", inplace=True)

adata.obs = meta.loc[adata.obs.index, :]

# 💡 view → copy (경고 방지)
adata.obs = adata.obs.copy()

# ✅ 5. 'label' 컬럼 생성 및 정수형 변환
adata.obs["label"] = adata.obs["disease__ontology_label"].apply(
    lambda x: 0 if x == "normal" else 1 if x == "COVID-19" else -1
).astype(int)

# ✅ 6. 라벨 없는 샘플 제거
adata = adata[adata.obs["label"] != -1]
adata.obs = adata.obs.copy()  # view 방지

# ✅ 7. cell type 컬럼 생성 (Detailed_Cell_Annotations → cell_type_annotation)
adata.obs["cell_type_annotation"] = adata.obs["Detailed_Cell_Annotations"]

# ✅ 8. donor_id → patient 명시적 복사
adata.obs["patient"] = adata.obs["donor_id"]

# ✅ 9. 1차 저장
adata.write_h5ad("covid.h5ad")

# ✅ 10. scGPT 임베딩 이후 처리 (singler_covid.csv 기반 cell type 대체)
adata = sc.read_h5ad("covid.h5ad")

ct = pd.read_csv("singler_covid.csv", index_col=0)

# 💡 cell type 결과 매핑
adata.obs["cell_type_annotation"] = ct.loc[adata.obs.index, "pruned.labels"]
adata = adata[adata.obs["cell_type_annotation"].notna()]

# ✅ 다시 patient 복사 (혹시 adata 교체됐을 경우를 위해)
adata.obs["patient"] = adata.obs["donor_id"]

# ✅ 최종 저장
adata.write_h5ad("../../covid.h5ad")

# ✅ 확인
print(adata.obs.columns)
print(adata.obs[["donor_id", "patient", "label", "cell_type_annotation"]].head())
