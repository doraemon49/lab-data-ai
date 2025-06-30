import scanpy as sc

adata = sc.read_h5ad("lupus_data/cellxgene_lupus.h5ad")

print(adata)
"""
AnnData object with n_obs × n_vars = 1263676 × 30867
    obs: 'library_uuid', 'assay_ontology_term_id', 'mapped_reference_annotation', 'is_primary_data', 'cell_type_ontology_term_id', 'author_cell_type', 'cell_state', 'sample_uuid', 'tissue_ontology_term_id', 'development_stage_ontology_term_id', 'disease_state', 'suspension_enriched_cell_types', 'suspension_uuid', 'suspension_type', 'donor_id', 'self_reported_ethnicity_ontology_term_id', 'organism_ontology_term_id', 'disease_ontology_term_id', 'sex_ontology_term_id', 'Processing_Cohort', 'ct_cov', 'ind_cov', 'tissue_type', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid'
    var: 'feature_is_filtered', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length', 'feature_type'
    uns: 'citation', 'default_embedding', 'schema_reference', 'schema_version', 'title'
    obsm: 'X_pca', 'X_umap'
"""
# 셀 수
print(adata.n_obs) # 1,263,676

# 유전자 수
print(adata.n_vars) # 30,867

# 셀 메타데이터 미리보기
print(adata.obs.head())
"""
                                                                            library_uuid  ... observation_joinid
index                                                                                     ...                   
CAAGGCCAGTATCGAA-1-1-0-0-0-0-0-0-0-0-0-0-0-0-0-...  3332f018-936e-4c4c-9105-99d9503db5a3  ...         a}Gz~P&pl&
CTAACTTCAATGAATG-1-1-0-0-0-0-0-0-0-0-0-0-0-0-0-...  70a004b7-4a17-4702-8910-4557aa0c4279  ...         tTQmA6ZGf`
AAGTCTGGTCTACCTC-1-1-0-0-0-0-0-0-0-0-0-0-0-0-0-...  152e2bfd-e9ea-4d70-a999-6f37fb3fb96c  ...         E-U&KcyH)S
GGCTCGATCGTTGACA-1-1-0-0-0-0-0-0-0-0-0-0-0-0-0-...  c2641f62-eb23-4dad-9c22-b52e72b79df2  ...         voUnoU!Ylu
ACACCGGCACACAGAG-1-1-0-0-0-0-0-0-0-0-0-0-0-0-0-...  222b358b-71e7-4b0f-9f9b-47b4c67aaa27  ...         _{EUND}(tY

[5 rows x 32 columns]
"""

# 유전자 메타데이터 미리보기
print(adata.var.head())
"""
                 feature_is_filtered       feature_name feature_reference feature_biotype feature_length    feature_type
ENSG00000243485                 True        MIR1302-2HG    NCBITaxon:9606            gene            623          lncRNA
ENSG00000237613                 True            FAM138A    NCBITaxon:9606            gene            888          lncRNA
ENSG00000186092                 True              OR4F5    NCBITaxon:9606            gene           2618  protein_coding
ENSG00000238009                 True  ENSG00000238009.6    NCBITaxon:9606            gene            629          lncRNA
ENSG00000239945                 True  ENSG00000239945.1    NCBITaxon:9606            gene           1319          lncRNA
"""

# 차원 축소 좌표
print(adata.obsm.keys()) # KeysView(AxisArrays with keys: X_pca, X_umap)

# 클러스터링 정보 등
print(adata.uns.keys()) # dict_keys(['citation', 'default_embedding', 'schema_reference', 'schema_version', 'title'])
