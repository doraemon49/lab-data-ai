# Download the following files from https://singlecell.broadinstitute.org/single_cell/study/SCP1303/
# 1. DCM_HCM_Expression_Matrix_raw_counts_V1.mtx
# 2. DCM_HCM_Expression_Matrix_genes_V1.tsv
# 3. DCM_HCM_Expression_Matrix_barcodes_V1.tsv
# 4. DCM_HCM_MetaData_V1.txt


from scipy.io import mmread
import scanpy as sc
import pandas as pd

data = mmread("DCM_HCM_Expression_Matrix_raw_counts_V1.mtx")
data = data.tocsr()

genes = pd.read_csv("DCM_HCM_Expression_Matrix_genes_V1.tsv", sep="\t", header=None).iloc[:,1].tolist()
genes = genes[:100]
data = data[:100, :]  # 유전자 개수 자르기
data = data.astype("float32")

barcodes = open( "DCM_HCM_Expression_Matrix_barcodes_V1.tsv").read().strip().split("\n")

meta = pd.read_csv("DCM_HCM_MetaData_V1.txt", sep="\t").drop(axis=0,index=0).reset_index(drop=True)

# adata = sc.AnnData(data)
# adata = sc.AnnData(data.T.tocsr())  # coo → csr 변환 후 AnnData 생성

# ✅ 'NAME' 컬럼을 인덱스로 설정하고, barcodes 순서로 정렬
meta.set_index("NAME", inplace=True)
meta = meta.loc[[bc for bc in barcodes if bc in meta.index]]  # ✅ barcodes 기준 필터링
barcodes = list(meta.index)  # ✅ 순서를 meta에 맞춤
data = data[:, [i for i, b in enumerate(barcodes)]]  # ✅ 열 순서 맞춤

# ✅ AnnData 생성
adata = sc.AnnData(data.T, obs=meta)


adata.obs.index = barcodes
adata.var.index = genes

sc.pp.filter_genes(adata, min_cells=5)

sc.pp.normalize_total(adata, target_sum=1e4)

sc.pp.log1p(adata)

# ------------------------------
# 1. 컬럼 존재 여부 확인하고 보정
# ------------------------------
if "donor_id" in adata.obs.columns:
    adata.obs.rename(columns={"donor_id": "patient"}, inplace=True)
else:
    adata.obs["patient"] = "unknown"

if "cell_type__ontology_label" in adata.obs.columns:
    adata.obs.rename(columns={"cell_type__ontology_label": "cell_type_annotation"}, inplace=True)
else:
    adata.obs["cell_type_annotation"] = "unknown"

# 강제 변환
adata.obs["label"] = adata.obs["disease__ontology_label"].map({
    "normal": 0,
    "hypertrophic cardiomyopathy": 1,
    "dilated cardiomyopathy": 2
})
# adata = adata[adata.obs["label"].notna()]
adata = adata[adata.obs["label"].notna()].copy()  # ✅ view → copy

adata.obs["label"] = adata.obs["label"].astype(int)

# Rename
adata.obs.rename(columns={
    "donor_id": "patient",
    "cell_type__ontology_label": "cell_type_annotation"
}, inplace=True)

# 강제 string 처리 (중요!)
adata.obs.columns = adata.obs.columns.astype(str)
adata.obs.index = adata.obs.index.astype(str)

# 저장
adata.write_h5ad("../../cardio.h5ad")  # 루트에 저장
print(adata.obs[["patient", "label", "cell_type_annotation"]].head())



# Extract the cell embeddings using the scGPT model (scgpt.tasks.embed_data) with the pretrained weights from the whole-human checkpoint. See https://github.com/bowang-lab/scGPT for instructions. 
# Store the embeddings in a new adata object and include the metadata of the old adata. Write the new adata object to the file cardio.h5ad .
#
# 저장
# adata.write_h5ad("../cardio.h5ad")

