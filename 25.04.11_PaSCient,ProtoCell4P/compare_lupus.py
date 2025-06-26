# python lupus_data/compare_lupus.py
"""
결론 : 유사한 데이터셋. 버전 차이인 듯. + scGPT에 학습된 데이터인 듯.
"""
import scanpy as sc
# download the following files from https://singlecell.broadinstitute.org/single_cell/study/SCP1289/
adata1 = sc.read_h5ad("lupus_data/lupus.h5ad")
# https://cellxgene.cziscience.com/collections/35d0b748-3eed-43a5-a1c4-1dade5ec5ca0
adata2 = sc.read_h5ad("lupus_data/cellxgene_lupus.h5ad")

# 1. Cell 이름 (.obs_names) 비교
# 두 데이터의 cell ID가 겹치는지 확인하면 가장 직접적으로 포함 여부를 알 수 있습니다.
# Cell 이름 집합으로 변환
cells1 = set(adata1.obs_names)
cells2 = set(adata2.obs_names)

# 공통 셀 개수
common_cells = cells1.intersection(cells2)
print(f"공통 cell 수: {len(common_cells)}")

# 비율로도 확인
print(f"lupus.h5ad의 셀 중 {len(common_cells) / len(cells1):.2%}가 cellxgene에 포함됨")
print(f"cellxgene의 셀 중 {len(common_cells) / len(cells2):.2%}가 lupus.h5ad에 포함됨")

# obs_names → 세포 이름 유사성 확인
# 앞 16자리만 추출해서 비교
cells1_short = set(adata1.obs_names.str[:16])
cells2_short = set(adata2.obs_names.str[:16])

common_short = cells1_short.intersection(cells2_short)
print(f"앞 16자리 기준 공통 셀 수: {len(common_short)}")  # 앞 16자리 기준 공통 셀 수: 479212
print("")



# 2. 개체 ID (ind_cov, donor_id) 비교
# 두 데이터 모두 개체(환자) 정보를 갖고 있으므로, ind_cov 또는 donor_id를 비교해서 겹치는 환자가 있는지 확인합니다.
ind1 = set(adata1.obs['ind_cov'])
# print("lupus donor : ",ind1) #  {'1480_1480', '901347200_901347200', '1731_1731', ... 이하 생략
print("lupus donor 수 : ", len(ind1)) # lupus donor 수 :  169

ind2 = set(adata2.obs['ind_cov']) if 'ind_cov' in adata2.obs else set(adata2.obs['donor_id'])
# print("cellxgene lupus donor : ",ind2) # {'FLARE001', '1480_1480', 'HC-503',  ... 이하 생략
print("cellxgene lupus donor 수 : ", len(ind2)) # cellxgene lupus donor 수 :  261

common_inds = ind1.intersection(ind2)
print(f"공통 개체 수: {len(common_inds)}") # 공통 개체 수: 83
print(f"lupus.h5ad의 개체 중 {len(common_inds) / len(ind1):.2%}가 cellxgene에 포함됨") # lupus.h5ad의 개체 중 49.11%가 cellxgene에 포함됨

# donor_id 앞 N글자 기준 공통 확인 코드
# N글자 기준 설정 (예: 5글자 또는 6글자)
N = 6
# lupus.h5ad donor: adata1.obs['ind_cov']
donor1_prefix = set([d[:N] for d in adata1.obs['ind_cov']])
# cellxgene donor: adata2.obs['donor_id']
donor2_prefix = set([d[:N] for d in adata2.obs['donor_id']])
# 공통 접두사 비교
common_prefix = donor1_prefix.intersection(donor2_prefix)

print(f"앞 {N}글자 기준 공통 donor 수: {len(common_prefix)}")                       # 앞 6글자 기준 공통 donor 수: 19
print(f"lupus.h5ad donor 중 {len(common_prefix) / len(donor1_prefix):.2%}가 겹침") # lupus.h5ad donor 중 13.77%가 겹침
print(f"cellxgene donor 중 {len(common_prefix) / len(donor2_prefix):.2%}가 겹침")   # cellxgene donor 중 8.76%가 겹침




# 3. 유전자 ID (var.index) 비교
# 유전자 수가 다르더라도, 유전자 ID가 겹치는지 살펴보면 분석 호환성이나 동일한 분석 파이프라인에서 왔는지를 알 수 있습니다.
genes1 = set(adata1.var.index)
genes2 = set(adata2.var.index)

common_genes = genes1.intersection(genes2)
print(f"공통 유전자 수: {len(common_genes)}")
print(f"lupus.h5ad 유전자 중 {len(common_genes) / len(genes1):.2%} 겹침")
print(f"cellxgene 유전자 중 {len(common_genes) / len(genes2):.2%} 겹침")

# var 유전자 이름 통일 → feature_name 또는 gene_symbol 기준 통일
# lupus.h5ad에서 gene symbol 추출
gene_symbols1 = set(adata1.var.index)

# cellxgene에서 gene symbol 필드 사용
gene_symbols2 = set(adata2.var['feature_name']) if 'feature_name' in adata2.var.columns else set()

# 교집합 확인
common_gene_symbols = gene_symbols1.intersection(gene_symbols2)
print(f"공통 gene symbol 수: {len(common_gene_symbols)}") # 공통 gene symbol 수: 2047
