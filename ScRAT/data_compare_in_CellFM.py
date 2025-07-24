"""
CellFM 학습 Data File Download
$ wget -O CellFM_data.zip "https://zenodo.org/records/15138665/files/CellFM_data.zip?download=1"
아래 명령어로 바로 압축을 해제할 수 있습니다 :
$ unzip CellFM_data.zip -d CellFM_data
"""

# python data_check.py
# COMBAT, Haniffa, SC4 데이터셋이 CellFM의 사전학습(pretraining)에 사용 되었는가?
# .h5ad 파일을 대상으로 하나씩 열어보며, 확인해보자.
# 확인 후 정리 > https://doraemin.tistory.com/254

import scanpy as sc

adata = sc.read_h5ad("CellFM_data/CellFM/LIHC_GSE140228.h5ad")
print("adata : ")
print(adata)
print("adata.obs : ")
print(adata.obs.head())
print("adata.var : ")
print(adata.var.head())

print(adata.obs.columns)  # 어떤 컬럼이 있는지
print("환자/Sample : ", adata.obs['Patient'].unique())  # 환자 ID가 몇 명인지

import scanpy as sc
import os

folder = "CellFM_data/CellFM"
h5ad_files = [f for f in os.listdir(folder) if f.endswith('.h5ad')]

# 관심 있는 obs 컬럼 키워드
keys_to_check = ['Patient', 'donor_id', 'sample_ID', 'study', 'GEO_Sample']

for filename in h5ad_files:
    path = os.path.join(folder, filename)
    print(f"\n📁 파일: {filename}")
    try:
        adata = sc.read_h5ad(path, backed='r')  # 메모리 절약 모드
        obs_keys = adata.obs.columns
        found = False
        for key in keys_to_check:
            if key in obs_keys:
                found = True
                try:
                    n_unique = adata.obs[key].nunique()
                    unique = adata.obs[key].unique()
                    print(f"  🔹 {key}: {n_unique} unique : {unique}")
                except Exception as e:
                    print(f"  ⚠️ {key} 컬럼은 있으나 처리 중 오류: {e}")
        if not found:
            print("  ⛔ 관련 obs 컬럼 없음")
        adata.file.close()
    except Exception as e:
        print(f"  ❌ 파일 열기 실패: {e}")


"""
adamson.h5ad >
AnnData object with n_obs × n_vars = 47795 × 1069
    obs: 'condition', 'pert_type', 'cell_type', 'cell_type_condition', 'guide identity', 'read count', 'UMI count', 'coverage', 'good coverage', 'number of cells', 'gene', 'dose_val', 'control', 'condition_name'
    var: 'gene_name', 'index', 'highly_variable', 'means', 'dispersions', 'dispersions_norm'
    uns: 'hvg', 'log1p', 'non_dropout_gene_idx', 'non_zeros_gene_idx', 'rank_genes_groups_cov_all', 'top_non_dropout_de_20', 'top_non_zero_de_20'

BCC_GSE123813.h5ad >
AnnData object with n_obs × n_vars = 52884 × 20638
    obs: 'UMAP_1', 'UMAP_2', 'Celltype (malignancy)', 'Celltype (major-lineage)', 'Celltype (minor-lineage)', 'Celltype (original)', 'Cluster', 'TimePoint', 'Sort', 'Celltype', 'Response', 'Patient', 'Source', 'Age', 'Gender', 'Stage', 'TNMstage', 'Treatment', 'cell_type'
    var: 'gene_ids', 'gene_names', 'feature_types', 'genome'
 🔹 Patient: 11 unique : ['su001', 'su002', 'su003', 'su004', 'su005', ..., 'su007', 'su008', 'su009', 'su010', 'su012']
Length: 11

Cell_Lines.h5ad >
AnnData object with n_obs × n_vars = 9531 × 32738
    obs: 'cell_type', 'batch'

DC.h5ad >
AnnData object with n_obs × n_vars = 576 × 26593
    obs: 'cell_type', 'batch'

Gene_classification.h5ad >
AnnData object with n_obs × n_vars = 35248 × 26390
    var: 'gene_label', 'dose_cond', 't1', 't2', 't3', 'train_t1', 'train_t2', 'train_t3', 'gene', 'old_name', 'new_name', 'origin'

Heart.h5ad >
AnnData object with n_obs × n_vars = 60668 × 27411
    obs: 'nCount_RNA', 'nFeature_RNA', 'age_group', 'cell_source', 'cell_states', 'sample', 'age.order', 'age.days.GA', 'size.CRL', 'size.NRL', 'stage', 'integration.groups', 'integrated_snn_res.0.1', 'clusters.low.res', 'clusters.high.res', 'clusters.res.2', 'clusters.res.3', 'condition', 'organism_ontology_term_id', 'tissue_ontology_term_id', 'assay_ontology_term_id', 'disease_ontology_term_id', 'cell_type_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'development_stage_ontology_term_id', 'sex_ontology_term_id', 'donor_id', 'suspension_type', 'is_primary_data', 'tissue_type', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid', 'batch'
    var: 'feature_is_filtered', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length'
    obsm: 'X_umap'
  🔹 donor_id: 13 unique : ['alexsc', 'BRC2252', 'BRC2256', 'BRC2260', 'BRC2262', ..., 'D4', 'D5', 'D6', 'D7', 'BRC2251']
Length: 13    

hPancreas_test.h5ad >
AnnData object with n_obs × n_vars = 4218 × 3000
    obs: 'Celltype'
    var: 'Gene Symbol'
    obsm: 'X_umap'

hPancreas_train.h5ad >
AnnData object with n_obs × n_vars = 10600 × 3000
    obs: 'Celltype'
    var: 'Gene Symbol'

hPancreas_test.h5ad >
AnnData object with n_obs × n_vars = 15476 × 33694
    obs: 'cell_type', 'batch'

Immune.h5ad >
AnnData object with n_obs × n_vars = 32484 × 12303
    obs: 'batch', 'chemistry', 'data_type', 'dpt_pseudotime', 'cell_type', 'mt_frac', 'n_counts', 'n_genes', 'sample_ID', 'size_factors', 'species', 'study', 'tissue'
    var: 'n_cells', 'gene', 'old_name', 'new_name'
    layers: 'counts'
  🔹 sample_ID: 4 unique : ['0', '1', '2', '3']
  🔹 study: 4 unique : ['Oetjen', 'Freytag', '10X', 'Sun']

LIHC_GSE140228.h5ad >
AnnData object with n_obs × n_vars = 61690 × 22143
    obs: 'UMAP_1', 'UMAP_2', 'Celltype (malignancy)', 'Celltype (major-lineage)', 'Celltype (minor-lineage)', 'Celltype (original)', 'Cluster', 'Source', 'Celltype_sub', 'Celltype_global', 'Sample', 'Histology', 'Tissue_sub', 'MajorCluster', 'Patient', 'Gender', 'Stage', 'TNMstage', 'cell_type'
    var: 'gene_ids', 'gene_names', 'feature_types', 'genome'
  🔹 Patient: 5 unique : ['D20171109', 'D20171215', 'D20180108', 'D20180110', 'D20180116']

Liver.h5ad >
AnnData object with n_obs × n_vars = 8444 × 20007
    obs: 'batch', 'Cell#', 'Cluster#', 'cell_type', 'n_genes'
    var: 'n_cells'

Lung.h5ad >
AnnData object with n_obs × n_vars = 10360 × 16327
    obs: 'nGene', 'nUMI', 'orig.ident', 'percent.mito', 'location', 'celltype', 'ID', 'GEO_Sample', 'cell_type', 'batch'
    var: 'feature'
  🔹 GEO_Sample: 4 unique : ['GSM3732850', 'GSM3732852', 'GSM3732854', 'GSM3732848']  

MCA.h5ad >
AnnData object with n_obs × n_vars = 6954 × 15006
    obs: 'batch', 'cell_type'

Myeloid.h5ad >
AnnData object with n_obs × n_vars = 13178 × 3000
    obs: 'cell_type', 'cancer_type', 'batch'
    obsm: 'X_pca', 'X_umap'

norman.h5ad >
AnnData object with n_obs × n_vars = 80506 × 1049
    obs: 'condition', 'pert_type', 'cell_type', 'guide_identity', 'read_count', 'UMI_count', 'coverage', 'gemgroup', 'good_coverage', 'number_of_cells', 'n_genes', 'cell_type_condition', 'lane', 'dose_val', 'control', 'condition_name', 'total_count'
    var: 'gene_name', 'highly_variable', 'means', 'dispersions', 'dispersions_norm'
    uns: 'hvg', 'log1p', 'non_dropout_gene_idx', 'non_zeros_gene_idx', 'rank_genes_groups_cov_all', 'top_non_dropout_de_20', 'top_non_zero_de_20'

Pancrm.h5ad >
AnnData object with n_obs × n_vars = 14767 × 15558
    obs: 'cell_type', 'batch'

PBMC_10K.h5ad> 
adata : 
AnnData object with n_obs × n_vars = 11990 × 3346
    obs: 'n_counts', 'batch', 'labels', 'str_labels', 'cell_type', 'train'
    var: 'n_counts-0', 'n_counts-1', 'n_counts'
    obsm: 'design', 'normalized_qc', 'qc_pc', 'raw_qc'

PBMC_368K.h5ad >
AnnData object with n_obs × n_vars = 4638 × 14236
    obs: 'batch', 'celltype', 'cell_type'

PBMC.h5ad >
AnnData object with n_obs × n_vars = 18868 × 6998
    obs: 'condition', 'n_counts', 'n_genes', 'mt_frac', 'cell_type', 'batch'
    var: 'gene_symbol', 'n_cells'
    obsm: 'X_pca', 'X_tsne', 'X_umap'

Skin.h5ad > 
AnnData object with n_obs × n_vars = 15457 × 30867
    obs: 'age', 'tissue_ontology_term_id', 'assay_ontology_term_id', 'disease_ontology_term_id', 'cell_type_ontology_term_id', 'self_reported_ethnicity_ontology_term_id', 'development_stage_ontology_term_id', 'sex_ontology_term_id', 'organism_ontology_term_id', 'is_primary_data', 'donor_id', 'suspension_type', 'Cluster', 'Celltype', 'tissue_type', 'sample_id', 'library_id', 'library_preparation_batch', 'library_sequencing_run', 'alignment_software', 'manner_of_death', 'sample_source', 'sample_collection_method', 'institute', 'sampled_site_condition', 'sample_preservation_method', 'sequenced_fragment', 'reference_genome', 'cell_enrichment', 'gene_annotation_version', 'cell_type', 'assay', 'disease', 'organism', 'sex', 'tissue', 'self_reported_ethnicity', 'development_stage', 'observation_joinid', 'batch'
    var: 'feature_is_filtered', 'feature_name', 'feature_reference', 'feature_biotype', 'feature_length'
    obsm: 'X_pca', 'X_umap'
🔹 donor_id: 5 unique : ['S2', 'S3', 'S4', 'S5', 'S1']
"""
