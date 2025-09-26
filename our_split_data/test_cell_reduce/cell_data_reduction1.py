import scanpy as sc
import numpy as np
import pandas as pd

# 재현성 고정
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# for repeat in range(5):
#     for fold in range(5):
#         print(f"🔁 Repeat {repeat}, Fold {fold}")

# adata = sc.read_h5ad('/data/project/kim89/0804_data/repeat_0/fold_0_test.h5ad')

# print(adata)
# print("✅ obs columns:", adata.obs.columns.tolist())

# ---------------------------
# 1) 전체에서 무작위로 N개만 남기기
# ---------------------------
def downsample_global(adata, n_keep=None, frac=None, random_state=RANDOM_SEED):
    """
    - n_keep: 남길 셀 개수 (정수). frac과 동시에 쓰지 말 것.
    - frac: 남길 비율 (0~1). n_keep과 동시에 쓰지 말 것.
    """
    assert (n_keep is None) ^ (frac is None), "n_keep 또는 frac 중 하나만 지정하세요."

    n_total = adata.n_obs
    if frac is not None:
        n_keep = max(1, int(round(n_total * frac)))

    n_keep = min(n_keep, n_total)  # 과도 지정 방지
    idx = rng.choice(n_total, size=n_keep, replace=False)
    idx = np.sort(idx)
    adata_ds = adata[idx].copy()
    print(f"🔻 Global downsample: {n_total} → {adata_ds.n_obs} cells")
    return adata_ds

# 예시: 전체의 20%만 남기기
# adata_small = downsample_global(adata, frac=0.2)

# 예시: 정확히 5,000개만 남기기
# adata_small = downsample_global(adata, n_keep=5000)


# ------------------------------------------
# 2) 특정 컬럼별(층화)로 균형 있게 샘플링
# ------------------------------------------
def downsample_by_group(adata, group_col, per_group=None, frac_per_group=None, min_per_group=1,
                        include_nan=False, random_state=RANDOM_SEED):
    """
    - group_col: obs의 그룹 기준 컬럼명 (예: 'manual_annotation', 'donor_id')
    - per_group: 각 그룹별로 남길 '개수'. (정수)
    - frac_per_group: 각 그룹별로 남길 '비율'(0~1). per_group와 동시에 쓰지 말 것.
    - min_per_group: 각 그룹에서 최소 보존할 개수(그룹 크기보다 클 경우 그룹 크기로 자동 조정)
    - include_nan: 그룹 라벨이 NaN인 셀을 포함할지 여부
    """
    assert (per_group is None) ^ (frac_per_group is None), "per_group 또는 frac_per_group 중 하나만 지정하세요."
    if group_col not in adata.obs.columns:
        raise ValueError(f"'{group_col}' 컬럼이 obs에 없습니다.")

    # NaN 그룹 처리
    obs = adata.obs[[group_col]].copy()
    if include_nan:
        grp_vals = obs[group_col].astype(object)  # NaN 유지
    else:
        mask = obs[group_col].notna()
        obs = obs[mask]
        grp_vals = obs[group_col]

    groups = grp_vals.astype("category")
    df = pd.DataFrame({"group": groups, "idx": np.arange(adata.n_obs)})
    if not include_nan:
        df = df.dropna(subset=["group"])

    keep_indices = []
    for g, sub in df.groupby("group", observed=True):
        n_g = len(sub)
        if frac_per_group is not None:
            k = int(round(n_g * frac_per_group))
        else:
            k = per_group

        # 최소/최대 한계
        k = max(min_per_group, k)
        k = min(k, n_g)

        chosen = sub.sample(n=k, random_state=random_state)["idx"].to_numpy()
        keep_indices.append(chosen)

    keep_indices = np.concatenate(keep_indices) if len(keep_indices) > 0 else np.array([], dtype=int)
    keep_indices.sort()

    adata_ds = adata[keep_indices].copy()

    # 전/후 요약
    before = df["group"].value_counts().sort_index()
    after = pd.Series(groups.iloc[keep_indices].values).value_counts().sort_index()
    print(f"🔻 Stratified downsample by '{group_col}': {adata.n_obs} → {adata_ds.n_obs} cells")
    print("📊 Before per-group:\n", before)
    print("📊 After  per-group:\n", after)
    return adata_ds

# 예시 A: manual_annotation별로 각 그룹에서 최대 300개씩만 유지
# adata_small = downsample_by_group(adata, group_col="manual_annotation", per_group=300, min_per_group=10)

# 예시 B: manual_annotation별로 각 그룹에서 30%만 유지(최소 10개는 보존)
# adata_small = downsample_by_group(adata, group_col="manual_annotation", frac_per_group=0.3, min_per_group=10)

# 예시 C: donor_id별로 각 도너에서 200개만 유지
# adata_small = downsample_by_group(adata, group_col="donor_id", per_group=200, min_per_group=50)

import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import sparse

# ===== 공통 유틸 =====
def _to_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)

def _rowwise_mean(X):
    # X: (n, g) dense or sparse
    if sparse.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel()
    else:
        return X.mean(axis=0)

def _group_mean_indices(n, chunk_size):
    # 0..n-1를 chunk_size 단위로 나눌 때 인덱스 묶음 리스트
    idx = np.arange(n)
    return [idx[i:i+chunk_size] for i in range(0, n, chunk_size)]

# ===== 방법 A: KMeans + medoid(centroid에 가장 가까운 실제 셀) 추출 =====
def reduce_by_kmeans_medoid(
    adata,
    n_targets=1000,
    use_space="pca",        # "pca" | obsm 키 | "gene"
    n_pcs=50,
    random_state=42
):
    """
    - n_targets: 최종 남길 셀 개수(k-means의 k)
    - use_space:
        "pca"  → adata.X로 PCA n_pcs 구한 뒤 그 공간에서 KMeans
        "gene" → 원래 유전자 공간(adata.X)에서 KMeans (고차원/무거울 수 있음)
        그 외  → adata.obsm[use_space]를 사용 (예: 'X_scGPT', 'scace_emb_1.0')
    - 각 클러스터에서 centroid에 가장 가까운 셀(유클리드 거리 최소)을 1개 선택
    """
    assert n_targets >= 1, "n_targets는 1 이상이어야 합니다."
    print(f"▶ KMeans+medoid: target={n_targets}, space={use_space}")

    # 1) 임베딩 준비
    if use_space == "pca":
        # PCA는 scanpy의 pca 결과를 재사용(없으면 새로 계산)
        if "X_pca" not in adata.obsm_keys():
            sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True if "highly_variable" in adata.var.columns else None)
        Z = adata.obsm["X_pca"][:, :n_pcs]
    elif use_space == "gene":
        Z = _to_dense(adata.X)
    else:
        if use_space not in adata.obsm_keys():
            raise ValueError(f"obsm['{use_space}'] 가 없습니다. obsm_keys={adata.obsm_keys()}")
        Z = np.asarray(adata.obsm[use_space])

    n = Z.shape[0]
    if n_targets > n:
        n_targets = n
        print(f"  (주의) n_targets가 셀 수보다 커서 {n_targets}로 조정되었습니다.")

    # 2) KMeans
    km = KMeans(n_clusters=n_targets, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Z)
    centers = km.cluster_centers_

    # 3) 각 클러스터의 medoid 셀 선택 (centroid에 가장 가까운 실제 셀 1개)
    chosen = []
    for k in range(n_targets):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        sub = Z[idx]
        # centroid와의 거리
        dists = np.linalg.norm(sub - centers[k], axis=1)
        medoid_local = np.argmin(dists)
        chosen.append(idx[medoid_local])

    chosen = np.unique(np.array(chosen, dtype=int))
    chosen.sort()
    adata_medoid = adata[chosen].copy()
    print(f"  ✅ 선택된 대표 셀: {adata_medoid.n_obs}개 (원본 {n}개)")
    return adata_medoid

# ===== 방법 B: 그룹 평균(pseudocell) =====
def reduce_by_group_mean(adata, group_col):
    """
    - group_col(예: 'manual_annotation', 'donor_id')마다 평균 발현 벡터를 만들어
      그룹당 1개 ‘평균 셀’을 생성합니다.
    - 결과는 실제 셀이 아니라 평균으로 만든 pseudo-cell(행=그룹)입니다.
    """
    if group_col not in adata.obs.columns:
        raise ValueError(f"'{group_col}' 컬럼이 obs에 없습니다.")
    groups = adata.obs[group_col].astype(str).values
    df = pd.DataFrame({"group": groups, "idx": np.arange(adata.n_obs)})

    X = adata.X  # (n_cells, n_genes)
    mean_rows = []
    new_obs = []

    for g, sub in df.groupby("group", observed=True):
        idx = sub["idx"].to_numpy()
        Xg = X[idx]
        mg = _rowwise_mean(Xg)  # (n_genes,)
        mean_rows.append(mg)
        # 대표 메타데이터: group 이름과 그룹 사이즈만 채워둠 (원하면 모드값/비율 추가 가능)
        new_obs.append({"group": g, "n_cells": len(idx)})

    X_mean = np.vstack(mean_rows)  # (n_groups, n_genes)
    obs_new = pd.DataFrame(new_obs).set_index(pd.Index([o["group"] for o in new_obs]))
    var_new = adata.var.copy()

    adata_group = sc.AnnData(X=X_mean, obs=obs_new, var=var_new)
    adata_group.obs_names = obs_new.index.astype(str)
    adata_group.obs[group_col] = adata_group.obs.index
    print(f"▶ 그룹 평균: 그룹 수 {adata_group.n_obs}개 (원본 셀 {adata.n_obs}개 → 평균화)")
    return adata_group

# ===== (옵션) 그룹 내 균등 분할 후 평균: 그룹당 여러 개의 pseudocell 만들기 =====
def reduce_by_group_chunked_mean(adata, group_col, chunk_size=100):
    """
    - group_col로 그룹화한 뒤, 각 그룹을 chunk_size로 나눠
      chunk마다 평균을 내어 여러 개의 pseudocell을 만듭니다.
    - 예) group_col='donor_id', chunk_size=200 → 도너별로 200개씩 평균 묶음 생성
    """
    if group_col not in adata.obs.columns:
        raise ValueError(f"'{group_col}' 컬럼이 obs에 없습니다.")
    groups = adata.obs[group_col].astype(str).values
    df = pd.DataFrame({"group": groups, "idx": np.arange(adata.n_obs)})

    X = adata.X
    mean_rows = []
    new_obs = []
    for g, sub in df.groupby("group", observed=True):
        idx = sub["idx"].to_numpy()
        chunks = _group_mean_indices(len(idx), chunk_size)
        for j, ch in enumerate(chunks):
            sel = idx[ch]
            mg = _rowwise_mean(X[sel])
            mean_rows.append(mg)
            new_obs.append({"group": g, "chunk_id": j, "n_cells": len(sel)})

    X_mean = np.vstack(mean_rows)
    obs_new = pd.DataFrame(new_obs)
    obs_new.index = [f"{r['group']}_chunk{r['chunk_id']}" for _, r in obs_new.iterrows()]
    var_new = adata.var.copy()

    adata_chunked = sc.AnnData(X=X_mean, obs=obs_new, var=var_new)
    print(f"▶ 그룹-분할 평균: {adata_chunked.n_obs}개 pseudocell 생성 (원본 {adata.n_obs}개)")
    return adata_chunked

# ======================
# 실제 호출 예시 (당신 데이터)
# ======================

# [예시 1] KMeans + medoid: 최종 2,000개 셀만 남기되, PCA 공간에서 대표 셀 추출
# adata_small = reduce_by_kmeans_medoid(adata, n_targets=2000, use_space="pca", n_pcs=50)

# [예시 2] 이미 있는 클러스터 라벨(예: 'merged_cluster_1.0')로 그룹 평균 → 그룹당 1개 셀
# adata_small = reduce_by_group_mean(adata, group_col="merged_cluster_1.0")

# [예시 3] 도너별로 200개씩 묶어 평균 → 도너 x (셀수/200) 개의 pseudocell 생성
# adata_small = reduce_by_group_chunked_mean(adata, group_col="donor_id", chunk_size=200)

# 저장 (원하실 때)
# adata_small.write('/data/project/kim89/0804_data/repeat_0/fold_0_test_reduced.h5ad')

# ---------------------------
# 결과 저장 (옵션)
# ---------------------------
for repeat in range(5):
    for fold in range(5):
        print(f"🔁 Repeat {repeat}, Fold {fold}")

        adata_train = sc.read_h5ad(f"/data/project/kim89/0804_data/repeat_{repeat}/fold_{fold}_train.h5ad")
        adata_test = sc.read_h5ad(f"/data/project/kim89/0804_data/repeat_{repeat}/fold_{fold}_test.h5ad")

        # 실행
        adata_small = reduce_by_group_mean(adata_train, group_col="manual_annotation")
        sc.write(f"/data/project/kim89/0804_data_cell_reduce/repeat_{repeat}/fold_{fold}_test.h5ad", adata_small)

        adata_small = reduce_by_group_mean(adata_test, group_col="manual_annotation")
        sc.write(f"/data/project/kim89/0804_data_cell_reduce/repeat_{repeat}/fold_{fold}_test.h5ad", adata_small)

