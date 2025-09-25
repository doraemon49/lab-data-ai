import anndata as ad
import pandas as pd

# 1. .h5ad 파일 불러오기
adata = ad.read_h5ad('parkinson.h5ad') 

# 2. 기본 정보 출력
print(adata)  # AnnData object summary (n_obs × n_vars, metadata 등)
"""
AnnData object with n_obs × n_vars = 2096155 × 17267
    obs: 'n_genes', 'n_counts', 'Brain_bank', 'RIN', 'path_braak_lb', 'derived_class2', 'PMI', 'organism_ontology_term_id', 'tissue_ontology_term_id', 'tissue_type', 'assay_ontology_term_id', 'disease_ontology_term_id', 'cell_type_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'development_stage_ontology_term_id', 'sex_ontology_term_id', 'donor_id', 'suspension_type', 'is_primary_data', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid'
    var: 'gene_name', 'n_cells', 'feature_is_filtered', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length', 'feature_type'
    uns: 'batch_condition', 'citation', 'genome', 'schema_reference', 'schema_version', 'title', 'uid'
    obsm: 'X_umap'
"""
print(f"관측치(Cells) 수: {adata.n_obs}")           # 관측치(Cells) 수: 2096155
print(f"변수(Genes) 수: {adata.n_vars}")            # 변수(Genes) 수: 17267           
print("obs 컬럼:", adata.obs.columns.tolist())      # obs 컬럼: ['n_genes', 'n_counts', 'Brain_bank', 'RIN', 'path_braak_lb', 'derived_class2', 'PMI', 'organism_ontology_term_id', 'tissue_ontology_term_id', 'tissue_type', 'assay_ontology_term_id', 'disease_ontology_term_id', 'cell_type_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'development_stage_ontology_term_id', 'sex_ontology_term_id', 'donor_id', 'suspension_type', 'is_primary_data', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid']
print("var 컬럼:", adata.var.columns.tolist())      # var 컬럼: ['gene_name', 'n_cells', 'feature_is_filtered', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length', 'feature_type']
print("layers 키 목록:", list(adata.layers.keys())) # layers 키 목록: []

# 3. 일부 데이터 확인
print("\n--- obs (첫 5개) ---")
print(adata.obs.head())

print("\n--- var (첫 5개) ---")
print(adata.var.head())

# 4. 표현값(X) 일부를 DataFrame으로 보기 (5개 셀 × 5개 유전자)
#    희소행렬(sparse)일 수 있으니 toarray() 이용
subset = pd.DataFrame(
    adata.X[:5, :5].toarray() if hasattr(adata.X, "toarray") else adata.X[:5, :5],
    index=adata.obs_names[:5],
    columns=adata.var_names[:5]
)
print("\n--- X 데이터 일부 (5×5) ---")
print(subset)
