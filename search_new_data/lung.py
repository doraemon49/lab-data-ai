import anndata as ad
import pandas as pd

# 1. .h5ad 파일 불러오기
adata = ad.read_h5ad('lung.h5ad') 

# 2. 기본 정보 출력
print(adata)  # AnnData object summary (n_obs × n_vars, metadata 등)
"""
AnnData object with n_obs × n_vars = 2282447 × 56239
    obs: 'suspension_type', 'donor_id', 'is_primary_data', 'assay_ontology_term_id', 'cell_type_ontology_term_id', 'development_stage_ontology_term_id', 'disease_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'tissue_ontology_term_id', 'organism_ontology_term_id', 'sex_ontology_term_id', "3'_or_5'", 'BMI', 'age_or_mean_of_age_range', 'age_range', 'anatomical_region_ccf_score', 'ann_coarse_for_GWAS_and_modeling', 'ann_finest_level', 'ann_level_1', 'ann_level_2', 'ann_level_3', 'ann_level_4', 'ann_level_5', 'cause_of_death', 'core_or_extension', 'dataset', 'fresh_or_frozen', 'log10_total_counts', 'lung_condition', 'mixed_ancestry', 'original_ann_level_1', 'original_ann_level_2', 'original_ann_level_3', 'original_ann_level_4', 'original_ann_level_5', 'original_ann_nonharmonized', 'reannotation_type', 'sample', 'scanvi_label', 'sequencing_platform', 'smoking_status', 'study', 'subject_type', 'tissue_coarse_unharmonized', 'tissue_detailed_unharmonized', 'tissue_dissociation_protocol', 'tissue_level_2', 'tissue_level_3', 'tissue_sampling_method', 'total_counts', 'transf_ann_level_1_label', 'transf_ann_level_1_uncert', 'transf_ann_level_2_label', 'transf_ann_level_2_uncert', 'transf_ann_level_3_label', 'transf_ann_level_3_uncert', 'transf_ann_level_4_label', 'transf_ann_level_4_uncert', 'transf_ann_level_5_label', 'transf_ann_level_5_uncert', 'tissue_type', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid'
    var: 'feature_is_filtered', 'original_gene_symbols', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length', 'feature_type'
    uns: 'batch_condition', 'citation', 'default_embedding', 'schema_reference', 'schema_version', 'title'
    obsm: 'X_scanvi_emb', 'X_umap'
    layers: 'soupX'
    obsp: 'connectivities', 'distances'
"""
print(f"관측치(Cells) 수: {adata.n_obs}")               # 관측치(Cells) 수: 2282447
print(f"변수(Genes) 수: {adata.n_vars}")                # 변수(Genes) 수: 56239          
print("obs 컬럼:", adata.obs.columns.tolist())          # obs 컬럼: ['suspension_type', 'donor_id', 'is_primary_data', 'assay_ontology_term_id', 'cell_type_ontology_term_id', 'development_stage_ontology_term_id', 'disease_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'tissue_ontology_term_id', 'organism_ontology_term_id', 'sex_ontology_term_id', "3'_or_5'", 'BMI', 'age_or_mean_of_age_range', 'age_range', 'anatomical_region_ccf_score', 'ann_coarse_for_GWAS_and_modeling', 'ann_finest_level', 'ann_level_1', 'ann_level_2', 'ann_level_3', 'ann_level_4', 'ann_level_5', 'cause_of_death', 'core_or_extension', 'dataset', 'fresh_or_frozen', 'log10_total_counts', 'lung_condition', 'mixed_ancestry', 'original_ann_level_1', 'original_ann_level_2', 'original_ann_level_3', 'original_ann_level_4', 'original_ann_level_5', 'original_ann_nonharmonized', 'reannotation_type', 'sample', 'scanvi_label', 'sequencing_platform', 'smoking_status', 'study', 'subject_type', 'tissue_coarse_unharmonized', 'tissue_detailed_unharmonized', 'tissue_dissociation_protocol', 'tissue_level_2', 'tissue_level_3', 'tissue_sampling_method', 'total_counts', 'transf_ann_level_1_label', 'transf_ann_level_1_uncert', 'transf_ann_level_2_label', 'transf_ann_level_2_uncert', 'transf_ann_level_3_label', 'transf_ann_level_3_uncert', 'transf_ann_level_4_label', 'transf_ann_level_4_uncert', 'transf_ann_level_5_label', 'transf_ann_level_5_uncert', 'tissue_type', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid'] 
print("var 컬럼:", adata.var.columns.tolist())          # var 컬럼: ['feature_is_filtered', 'original_gene_symbols', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length', 'feature_type']
print("layers 키 목록:", list(adata.layers.keys()))     # layers 키 목록: ['soupX']

# 3. 일부 데이터 확인
print("\n--- obs (첫 5개) ---")
print(adata.obs.head())
"""
--- obs (첫 5개) ---
                                          suspension_type  ... observation_joinid
CGATGTAAGTTACGGG_SC10                                cell  ...         LL;0n*@mx8
cc05p_CATGCCTGTGTGCCTG_carraro_csmc                  cell  ...         )rNf~Q0&BX
ATTCTACCAAGGTTCT_HD68                                cell  ...         5%kv|ie@!5
D062_TGACCCTTCAAACCCA-sub_wang_sub_batch3         nucleus  ...         Jq?*-$kHDp
muc9826_GTCGTGAGAGGA_mayr                            cell  ...         m35M8pyQm_

[5 rows x 70 columns]
"""

print("\n--- var (첫 5개) ---")
print(adata.var.head())
"""
--- var (첫 5개) ---
                 feature_is_filtered  ...    feature_type
ENSG00000121410                False  ...  protein_coding
ENSG00000268895                False  ...          lncRNA
ENSG00000148584                False  ...  protein_coding
ENSG00000175899                False  ...  protein_coding
ENSG00000245105                False  ...          lncRNA

[5 rows x 7 columns]
"""
# 4. 표현값(X) 일부를 DataFrame으로 보기 (5개 셀 × 5개 유전자)
#    희소행렬(sparse)일 수 있으니 toarray() 이용
subset = pd.DataFrame(
    adata.X[:5, :5].toarray() if hasattr(adata.X, "toarray") else adata.X[:5, :5],
    index=adata.obs_names[:5],
    columns=adata.var_names[:5]
)
print("\n--- X 데이터 일부 (5×5) ---")
print(subset)
"""
--- X 데이터 일부 (5×5) ---
                                           ENSG00000121410  ...  ENSG00000245105
CGATGTAAGTTACGGG_SC10                                  0.0  ...              0.0
cc05p_CATGCCTGTGTGCCTG_carraro_csmc                    0.0  ...              0.0
ATTCTACCAAGGTTCT_HD68                                  0.0  ...              0.0
D062_TGACCCTTCAAACCCA-sub_wang_sub_batch3              0.0  ...              0.0
muc9826_GTCGTGAGAGGA_mayr                              0.0  ...              0.0

[5 rows x 5 columns]
"""