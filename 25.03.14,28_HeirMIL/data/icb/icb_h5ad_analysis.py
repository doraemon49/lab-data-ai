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

"""
<< 실행 결과 >>

(venv) kim89@ailab-System-Product-Name:~/hier-mil$ python data/icb/icb_h5ad_analysis.py
AnnData object with n_obs × n_vars = 9292 × 824
    obs: 'sample_id', 'cell_id', 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'biosample_id', 'species', 'species__ontology_label', 'disease', 'disease__ontology_label', 'organ', 'organ__ontology_label', 'library_preparation_protocol', 'library_preparation_protocol__ontology_label', 'sex', 'cell.type', 'flow', 'X', 'Y', 'Gender', 'Primary Location', 'Immunotherapy #1', 'Immunotherapy #2', 'Immunotherapy #3', 'Immunotherapy #4', 'Targeted Therapy (dates)', 'CNS metastasic sites', 'Systemic sites of metastasis', 'SNaPshot Mutations', 'Location of surgery #1', 'Surgery #1 Single-cell ID', 'Location of surgery #2', 'Surgery #2 Single-cell ID', 'Location of surgery #3', 'Surgery #3 Single-cell ID', 'Location of surgery #4', 'Surgery #4 Single-cell ID', 'Pre/post ICI', 'outcome', 'Presence of necrosis on H&E', 'Var.41', '.1', '.2', '.3', '.4', '.5', '.6', 'donor_id_prepost', 'donor_id_responder', 'enough_cells', 'donor_id_prepost_responder', 'pre_post', 'Study_name', 'Cancer_type', 'Primary_or_met', 'sample_id_pre_post', 'total_cell_per_patient', 'cell_type_for_count', 'total_T_Cell', 'normalized_CD8_totalcells', 'RNA_snn_res.0.8', 'seurat_clusters', 'treatment', 'sort', 'cluster', 'UMAP1', 'UMAP2', 'Tumor.Type', 'Treatment', 'Ongoing.Vismodegib.treatment', 'Prior.treatment', 'Response', 'Best...change', 'scRNA.pre.site', 'scRNA.days.pre.treatment', 'scRNA.post.site', 'scRNA.days.post.treatment', 'Adaptive.pre.site', 'Adaptive.days.pre.treatment', 'Adaptive.post.site', 'Adaptive.days.post.treatment', 'PBMC.Adaptive.days.pre.treatment', 'PBMC.Adaptive.days.post.treatment', 'Exome.pre.site', 'Exome.days.pre.treatment', 'Exome.post.site', 'Exome.days.post.treatment', 'epi', 'sample_id_outcome', 'cell_type_for_count.x', 'cell_type_for_count.y', 'total_T_Cell_only', 'normalized_CD8_actual_totalcells', 'cell.types', 'treatment.group', 'Cohort', 'no.of.genes', 'no.of.reads', 'NAME', 'LABELS', 'tumor', 'immune_outcome', 'Immune_resistance.up', 'Immune_resistance.down', 'OE.Immune_resistance', 'OE.Immune_resistance.up', 'OE.Immune_resistance.down', 'no.genes', 'log.no.reads', 'technology', 'n_cells', 'patient', 'age', 'smoking_status', 'PY', 'diagnosis_recurrence', 'disease_extent', 'AJCC_T', 'AJCC_N', 'AJCC_M', 'AJCC_stage', 'sample_primary_met', 'size', 'site', 'histology', 'genetic_hormonal_features', 'grade', 'KI67', 'chemotherapy_exposed', 'chemotherapy_response', 'targeted_rx_exposed', 'targeted_rx_response', 'ICB_exposed', 'ICB_response', 'ET_exposed', 'ET_response', 'time_end_of_rx_to_sampling', 'post_sampling_rx_exposed', 'post_sampling_rx_response', 'PFS_DFS', 'OS', 'total_T.CD8', 'timepoint', 'cellType', 'cohort', 'treatment_info', 'Cancer_type_pre_post', 'sample', 'id', 'Type', 'No.a', 'Sex', 'Age..Years.b', 'Race', 'Diagnosis', 'Stage', 'Etiology', 'Biopsy.Timingc', 'Treatmentd', 'Mode.of.Actione', 'set', 'Sample', 'Source', 'Stage.y', 'Mode.of.Actione_2', 'sample_id_Mode.of.Actione_2', 'ICB_Exposed', 'ICB_Response', 'TKI_Exposed', 'Initial_Louvain_Cluster', 'Lineage', 'InferCNV', 'FinalCellType', 'sex.x', 'cancer_type', 'sex.y', 'treated_naive', 'Cancer_type_update', 'Outcome', 'Combined_outcome', 'Malignant_clusters', 'patient_ID', 'pre_post_outcome', 'percent.mito', 'percent.ribo', 'pANN_0.25_0.21_50', 'DoubletFinder', 'pANN_0.25_0.21_642', 'pANN_0.25_0.21_61', 'pANN_0.25_0.21_7', 'pANN_0.25_0.21_18', 'pANN_0.25_0.21_94', 'pANN_0.25_0.21_6', 'pANN_0.25_0.21_35', 'Study_name_cancer', 'label', 'cell_type_annotation'
(9292, 824)
   HAVCR2  CTLA4  PDCD1  ...  EP300  PHF6  KRAS
0     0.0    0.0    0.0  ...    0.0   0.0   0.0
1     0.0    0.0    0.0  ...    0.0   0.0   0.0
2     0.0    0.0    0.0  ...    0.0   0.0   0.0
3     0.0    0.0    0.0  ...    0.0   0.0   0.0
4     0.0    0.0    0.0  ...    0.0   0.0   0.0

[5 rows x 824 columns]
==== column : 824개 유전자 ====
(824, 0)
Index(['HAVCR2', 'CTLA4', 'PDCD1', 'IDO1', 'CXCL10', 'CXCL9', 'HLA-DRA',
       'STAT1', 'IFNG', 'CD3E',
       ...
       'EZH2', 'TP53', 'CALR', 'STAG2', 'CEBPA', 'CUX1', 'U2AF1', 'EP300',
       'PHF6', 'KRAS'],
      dtype='object', length=824)
Empty DataFrame
Columns: []
Index: [HAVCR2, CTLA4, PDCD1, IDO1, CXCL10]
==== row : 9292개 세포의 정보 197가지 ====
(9292, 197)
Index(['sample_id', 'cell_id', 'orig.ident', 'nCount_RNA', 'nFeature_RNA',
       'biosample_id', 'species', 'species__ontology_label', 'disease',
       'disease__ontology_label', 'organ', 'organ__ontology_label',
       'library_preparation_protocol',
       'library_preparation_protocol__ontology_label', 'sex', 'cell.type',
       'flow', 'X', 'Y', 'Gender', 'Primary Location', 'Immunotherapy #1',
       'Immunotherapy #2', 'Immunotherapy #3', 'Immunotherapy #4',
       'Targeted Therapy (dates)', 'CNS metastasic sites',
       'Systemic sites of metastasis', 'SNaPshot Mutations',
       'Location of surgery #1', 'Surgery #1 Single-cell ID',
       'Location of surgery #2', 'Surgery #2 Single-cell ID',
       'Location of surgery #3', 'Surgery #3 Single-cell ID',
       'Location of surgery #4', 'Surgery #4 Single-cell ID', 'Pre/post ICI',
       'outcome', 'Presence of necrosis on H&E', 'Var.41', '.1', '.2', '.3',
       '.4', '.5', '.6', 'donor_id_prepost', 'donor_id_responder',
       'enough_cells'],
      dtype='object')
Index(['donor_id_prepost_responder', 'pre_post', 'Study_name', 'Cancer_type',
       'Primary_or_met', 'sample_id_pre_post', 'total_cell_per_patient',
       'cell_type_for_count', 'total_T_Cell', 'normalized_CD8_totalcells',
       'RNA_snn_res.0.8', 'seurat_clusters', 'treatment', 'sort', 'cluster',
       'UMAP1', 'UMAP2', 'Tumor.Type', 'Treatment',
       'Ongoing.Vismodegib.treatment', 'Prior.treatment', 'Response',
       'Best...change', 'scRNA.pre.site', 'scRNA.days.pre.treatment',
       'scRNA.post.site', 'scRNA.days.post.treatment', 'Adaptive.pre.site',
       'Adaptive.days.pre.treatment', 'Adaptive.post.site',
       'Adaptive.days.post.treatment', 'PBMC.Adaptive.days.pre.treatment',
       'PBMC.Adaptive.days.post.treatment', 'Exome.pre.site',
       'Exome.days.pre.treatment', 'Exome.post.site',
       'Exome.days.post.treatment', 'epi', 'sample_id_outcome',
       'cell_type_for_count.x', 'cell_type_for_count.y', 'total_T_Cell_only',
       'normalized_CD8_actual_totalcells', 'cell.types', 'treatment.group',
       'Cohort', 'no.of.genes', 'no.of.reads', 'NAME', 'LABELS'],
      dtype='object')
Index(['tumor', 'immune_outcome', 'Immune_resistance.up',
       'Immune_resistance.down', 'OE.Immune_resistance',
       'OE.Immune_resistance.up', 'OE.Immune_resistance.down', 'no.genes',
       'log.no.reads', 'technology', 'n_cells', 'patient', 'age',
       'smoking_status', 'PY', 'diagnosis_recurrence', 'disease_extent',
       'AJCC_T', 'AJCC_N', 'AJCC_M', 'AJCC_stage', 'sample_primary_met',
       'size', 'site', 'histology', 'genetic_hormonal_features', 'grade',
       'KI67', 'chemotherapy_exposed', 'chemotherapy_response',
       'targeted_rx_exposed', 'targeted_rx_response', 'ICB_exposed',
       'ICB_response', 'ET_exposed', 'ET_response',
       'time_end_of_rx_to_sampling', 'post_sampling_rx_exposed',
       'post_sampling_rx_response', 'PFS_DFS', 'OS', 'total_T.CD8',
       'timepoint', 'cellType', 'cohort', 'treatment_info',
       'Cancer_type_pre_post', 'sample', 'id', 'Type'],
      dtype='object')
Index(['No.a', 'Sex', 'Age..Years.b', 'Race', 'Diagnosis', 'Stage', 'Etiology',
       'Biopsy.Timingc', 'Treatmentd', 'Mode.of.Actione', 'set', 'Sample',
       'Source', 'Stage.y', 'Mode.of.Actione_2', 'sample_id_Mode.of.Actione_2',
       'ICB_Exposed', 'ICB_Response', 'TKI_Exposed', 'Initial_Louvain_Cluster',
       'Lineage', 'InferCNV', 'FinalCellType', 'sex.x', 'cancer_type', 'sex.y',
       'treated_naive', 'Cancer_type_update', 'Outcome', 'Combined_outcome',
       'Malignant_clusters', 'patient_ID', 'pre_post_outcome', 'percent.mito',
       'percent.ribo', 'pANN_0.25_0.21_50', 'DoubletFinder',
       'pANN_0.25_0.21_642', 'pANN_0.25_0.21_61', 'pANN_0.25_0.21_7',
       'pANN_0.25_0.21_18', 'pANN_0.25_0.21_94', 'pANN_0.25_0.21_6',
       'pANN_0.25_0.21_35', 'Study_name_cancer', 'label',
       'cell_type_annotation'],
      dtype='object')
                                                    sample_id  ... cell_type_annotation
Row.names                                                      ...                     
Breast_previous_Breast_BIOKEY_10_Pre_AAAGCAAAGC...  BIOKEY_10  ...      Mesangial cells
Breast_previous_Breast_BIOKEY_10_Pre_AAATGCCGTT...  BIOKEY_10  ...                  HSC
Breast_previous_Breast_BIOKEY_10_Pre_AACTCTTGTA...  BIOKEY_10  ...      Mesangial cells
Breast_previous_Breast_BIOKEY_10_Pre_AACTGGTAGT...  BIOKEY_10  ...              B-cells
Breast_previous_Breast_BIOKEY_10_Pre_AACTTTCAGG...  BIOKEY_10  ...           Adipocytes

[5 rows x 197 columns]
57
23
2
"""