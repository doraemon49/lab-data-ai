import scanpy as sc

adata = sc.read_h5ad("lupus_data/lupus.h5ad")

print(adata)
"""
AnnData object with n_obs × n_vars = 834096 × 32738
    obs: 'disease_cov', 'ct_cov', 'pop_cov', 'ind_cov', 'well', 'batch_cov', 'batch'
    var: 'gene_ids-0-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0-0', 'gene_ids-1-0-0-0-0-0', 'gene_ids-1-0-0-0-0', 'gene_ids-1-0-0-0', 'gene_ids-1-0-0', 'gene_ids-1-0', 'gene_ids-1'
"""
# 셀 수
print(adata.n_obs) # 834,096

# 유전자 수
print(adata.n_vars) # 32,738

# 셀 메타데이터 미리보기
print(adata.obs.head())
"""
                                             disease_cov           ct_cov pop_cov              ind_cov       well  batch_cov batch
index                                                                                                                             
AAACCTGAGAGCAATT-1-0-0-0-0-0-0-0-0-0-0-0-0-0         sle      CD4 T cells   WHITE            1760_1760  YE_8-16-1  lupus8.16     0
AAACCTGAGCAATATG-1-0-0-0-0-0-0-0-0-0-0-0-0-0         sle   Megakaryocytes   WHITE  901560200_901560200  YE_8-16-1  lupus8.16     0
AAACCTGAGTCCCACG-1-0-0-0-0-0-0-0-0-0-0-0-0-0         sle      CD4 T cells   ASIAN            1584_1584  YE_8-16-1  lupus8.16     0
AAACCTGAGTGCAAGC-1-0-0-0-0-0-0-0-0-0-0-0-0-0         sle      CD8 T cells   ASIAN            1597_1597  YE_8-16-1  lupus8.16     0
AAACCTGCAAGGCTCC-1-0-0-0-0-0-0-0-0-0-0-0-0-0         sle  CD14+ Monocytes   WHITE            1775_1775  YE_8-16-1  lupus8.16     0
"""

# 유전자 메타데이터 미리보기
print(adata.var.head())
"""
             gene_ids-0-0-0-0-0-0-0-0-0-0-0-0-0 gene_ids-1-0-0-0-0-0-0-0-0-0-0-0-0  ...     gene_ids-1-0       gene_ids-1
index                                                                               ...                                  
MIR1302-10                      ENSG00000243485                    ENSG00000243485  ...  ENSG00000243485  ENSG00000243485
FAM138A                         ENSG00000237613                    ENSG00000237613  ...  ENSG00000237613  ENSG00000237613
OR4F5                           ENSG00000186092                    ENSG00000186092  ...  ENSG00000186092  ENSG00000186092
RP11-34P13.7                    ENSG00000238009                    ENSG00000238009  ...  ENSG00000238009  ENSG00000238009
RP11-34P13.8                    ENSG00000239945                    ENSG00000239945  ...  ENSG00000239945  ENSG00000239945

[5 rows x 14 columns]
"""

# 차원 축소 좌표
print(adata.obsm.keys()) # KeysView(AxisArrays with keys: )

# 클러스터링 정보 등
print(adata.uns.keys()) # dict_keys([])
