"""
ver2. 코드 흐름은 n×n 유사도(밀집 행렬) 계산 → precomputed 거리로 HCA라서, 셀 수가 조금만 커져도 시간·메모리 폭발이 납니다.

핵심 변경점 (요약)

1. 유사도 희소화:
U = sign(X - mean)까지는 동일
NearestNeighbors(metric="cosine")로 **kNN 그래프(희소)**만 계산
→ n×n 밀집행렬 대신 O(n·k) 메모리/연산

2. 클러스터링 교체:
HCA(precomputed 거리) → Leiden(그래프 군집)
Scanpy의 sc.tl.leiden은 희소 kNN 그래프에 최적화

3. 즉시 저장/체크포인트:
반복/폴드마다 대표 pseudo-cell을 바로 .h5ad로 저장
진행률/타이밍 출력

4. 테스트 시간 단축 옵션:
n_hvg, n_neighbors, k_range 등 하이퍼파라미터를 합리적으로 축소
필요시 train 기준 HVG/스케일러를 test에 재사용(재현성 + 속도)
"""

import os, time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

# pip3 install igraph
# pip3 install leidenalg

# python data_cell_reduce2-2.py

# ----- 빠른 Corr-approx: kNN on U (sparse graph) + Leiden -----
def build_corr_knn_graph(X, n_neighbors=30):
    """
    X: (n_cells, n_genes) float32, scaled
    1) U = sign(X - global_mean)
    2) cosine kNN (행 중심화/정규화는 cosine으로 충분)
    return: sparse adjacency (symmetric), where weight ~ similarity
    """
    gmean = X.mean(axis=0, keepdims=True)
    U = np.sign(X - gmean).astype(np.float32)

    # kNN in U-space with cosine distance
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    nn.fit(U)
    dists, idxs = nn.kneighbors(U, return_distance=True)  # d in [0,2], cosine
    # similarity = 1 - distance
    sims = 1.0 - dists
    n = U.shape[0]

    # build sparse directed graph, then symmetrize (max)
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = idxs.ravel()
    data = sims.ravel()

    G = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # ensure self-loops (needed by Leiden sometimes)
    G = G.maximum(G.T)
    G.setdiag(1.0)
    G.eliminate_zeros()
    return G

def cluster_leiden_from_graph(adata, G, resolution=1.0):
    """
    Plug user graph into AnnData and run Leiden.
    """
    # neighbors slots
    adata.obsp["connectivities"] = G
    # distance is optional; we can approximate as 1 - w
    adata.obsp["distances"] = sp.csr_matrix(G.shape, dtype=np.float32)
    sc.tl.leiden(adata, resolution=resolution, key_added="corr_leiden")

def cluster_representatives(X, labels, method='mean'):
    reps = {}
    uniq = np.unique(labels)
    for k in uniq:
        idx = np.where(labels == k)[0]
        Xk = X[idx]
        if method == 'mean':
            reps[k] = Xk.mean(axis=0)
        elif method == 'sum':
            reps[k] = Xk.sum(axis=0)
        elif method == 'medoid':
            # 간단한 medoid (유클리드)
            from sklearn.metrics import pairwise_distances
            dk = pairwise_distances(Xk, Xk, metric='euclidean')
            reps[k] = Xk[np.argmin(dk.mean(axis=1))]
        else:
            raise ValueError("method must be one of {'mean','sum','medoid'}")
    return reps

def run_fast_corr_pipeline(
    adata, n_hvg=2000, n_neighbors=30, resolution=1.0, rep_method='mean'
):
    ad = adata.copy()
    # 표준 전처리 (존재하면 생략됨)
    if 'log1p' not in ad.uns.get('pp', {}):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)

    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg, flavor='seurat_v3')
    ad = ad[:, ad.var['highly_variable']].copy()

    X = ad.X.A if hasattr(ad.X, 'A') else ad.X
    X = np.asarray(X, dtype=np.float32)

    # gene-wise scaling: ANOVA는 안 쓰지만, cosine kNN 안정화에 도움
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X).astype(np.float32)

    # 희소 그래프 구성
    G = build_corr_knn_graph(X, n_neighbors=n_neighbors)

    # Leiden 군집
    cluster_leiden_from_graph(ad, G, resolution=resolution)
    labels = ad.obs['corr_leiden'].astype(str).values

    # 대표값 (pseudo-cell)
    reps = cluster_representatives(X, labels, method=rep_method)

    return dict(
        labels=labels,
        representatives=reps,
        n_clusters=len(np.unique(labels)),
        hvg_mask=np.asarray(ad.var_names)
    )

import numpy as np
import pandas as pd
import scanpy as sc
from collections import Counter

def majority_vote(series: pd.Series):
    """범주형 최빈값과 비율 반환"""
    cnt = series.value_counts(dropna=True)
    if len(cnt) == 0:
        return np.nan, np.nan
    mode = cnt.index[0]
    frac = float(cnt.iloc[0]) / float(series.notna().sum())
    return mode, frac

def save_pseudocells_with_metadata(
    adata_orig: sc.AnnData,
    labels: np.ndarray,          # 길이 = adata_orig.n_obs
    reps_dict: dict,             # {cluster_id: vector}
    gene_names: pd.Index,        # reps_dict 벡터 순서에 대응하는 var_names (HVG)
    out_h5ad_path: str,
    mapping_csv_path: str = None,
    obs_cols_to_summarize: list = None,   # 요약하고 싶은 원본 obs 컬럼들
):
    """
    - adata_orig: 원본 AnnData (HVG 선택/스케일 전 단계 혹은 같은 HVG 순서여도 무방)
    - labels: 각 셀의 클러스터 라벨 (문자/정수 상관없음)
    - reps_dict: cluster_id -> 대표 벡터 (len = len(gene_names))
    - gene_names: 대표 벡터의 유전자 인덱스
    - obs_cols_to_summarize: None이면 원본의 주요 컬럼 일부 자동 선택 예시 제공
    """
    # 정렬된 클러스터 순서
    ks = sorted(reps_dict, key=lambda x: int(x) if str(x).isdigit() else str(x))
    mat = np.vstack([reps_dict[k] for k in ks]).astype(np.float32)

    # --- obs 요약 메타데이터 만들기 ---
    obs_df = pd.DataFrame(index=[f"cluster_{k}" for k in ks])
    obs_df["n_cells"] = 0

    # 요약 대상 컬럼 지정 (없으면 예시로 몇 개 자동 선택)
    if obs_cols_to_summarize is None:
        # 원본에 존재하면 좋은 것들 예시
        candidates = [
            "donor_id", "patient", "manual_annotation", "singler_annotation",
            "Coarse_Cell_Annotations", "Detailed_Cell_Annotations",
            "disease__ontology_label", "organ__ontology_label", "sex", "age"
        ]
        obs_cols_to_summarize = [c for c in candidates if c in adata_orig.obs.columns]

    # 각 클러스터별 원본 인덱스 수집
    labels = pd.Series(labels, index=adata_orig.obs_names, name="cluster")
    clust_to_cells = {k: labels.index[labels.astype(str) == str(k)] for k in ks}

    # 매핑 CSV (원본 셀 -> 클러스터)
    if mapping_csv_path is not None:
        map_df = pd.DataFrame({
            "cell": labels.index,
            "cluster": labels.values
        })
        map_df.to_csv(mapping_csv_path, index=False)

    # 숫자형/범주형 나눠서 요약
    for k in ks:
        cell_ids = clust_to_cells[k]
        obs_df.loc[f"cluster_{k}", "n_cells"] = len(cell_ids)

        if len(cell_ids) == 0:
            continue

        sub = adata_orig.obs.loc[cell_ids]

        for col in obs_cols_to_summarize:
            if col not in sub.columns:
                continue
            s = sub[col]

            if pd.api.types.is_numeric_dtype(s):
                obs_df.loc[f"cluster_{k}", f"mean_{col}"] = float(np.nanmean(s.values))
            else:
                mode, frac = majority_vote(s.astype("object"))
                obs_df.loc[f"cluster_{k}", f"mode_{col}"] = mode
                obs_df.loc[f"cluster_{k}", f"mode_{col}_frac"] = frac

    # --- var 메타데이터 구성 ---
    var_df = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
    # 원본 var에 같은 이름이 있으면 몇 개 필드를 복사 (예: 'highly_variable', 'means' 등)
    for col in ["highly_variable", "means", "dispersions", "dispersions_norm", "Gene", "id_in_vocab", "n_cells"]:
        if col in adata_orig.var.columns:
            common = var_df.index.intersection(adata_orig.var.index)
            var_df.loc[common, col] = adata_orig.var.loc[common, col]

    # --- AnnData 작성 및 저장 ---
    ad_pseudo = sc.AnnData(
        X=mat,
        obs=obs_df,
        var=var_df
    )
    ad_pseudo.uns["pseudo_info"] = {
        "note": "Corr-based pseudo-cells with summarized metadata",
        "n_original_cells": int(adata_orig.n_obs),
        "n_original_genes": int(adata_orig.n_vars),
        "n_clusters": int(len(ks)),
    }
    ad_pseudo.write(out_h5ad_path)

# -------------------- 5x5 루프 + 체크포인트 저장 --------------------
base = "/data/project/kim89/0804_data"
save = "/data/project/kim89/0804_data_cell_reduce"
out_tag = "corr_sparse"  # 출력 파일명 태그
os.makedirs(base, exist_ok=True)

# def save_pseudocells(reps_dict, gene_names, out_path):
#     # reps_dict: {cluster_id: vector}
#     ks = sorted(reps_dict, key=lambda x: int(x) if str(x).isdigit() else str(x))
#     mat = np.vstack([reps_dict[k] for k in ks]).astype(np.float32)
#     obs = pd.DataFrame(index=[f"cluster_{k}" for k in ks])
#     var = pd.DataFrame(index=gene_names)
#     ad = sc.AnnData(mat, obs=obs, var=var)
#     ad.write(out_path)

for repeat in range(5):
    for fold in range(5):
        t0 = time.time()
        print(f"\n🔁 Repeat {repeat}, Fold {fold}")
        tr_path = os.path.join(base, f"repeat_{repeat}", f"fold_{fold}_train.h5ad")
        te_path = os.path.join(base, f"repeat_{repeat}", f"fold_{fold}_test.h5ad")

        adata_train = sc.read_h5ad(tr_path)
        adata_test  = sc.read_h5ad(te_path)

        # --- TRAIN ---
        print("  ▶ TRAIN: build graph & cluster ...")
        train_out = run_fast_corr_pipeline(
            adata_train,
            n_hvg=1500,           # 더 빠르게
            n_neighbors=30,
            resolution=1.0,
            rep_method='mean'
        )
        # 즉시 저장 (체크포인트)
        out_dir = os.path.join(save, f"repeat_{repeat}")
        os.makedirs(out_dir, exist_ok=True)
        # train_outfile = os.path.join(out_dir, f"fold_{fold}_train_{out_tag}_pseudocells.h5ad")
        # save_pseudocells(train_out['representatives'], train_out['hvg_mask'], train_outfile)
        # print(f"  ✅ saved TRAIN pseudo-cells -> {train_outfile} (k={train_out['n_clusters']})")
        # 기존 save_pseudocells(...) 대신:
        
        train_outfile = os.path.join(out_dir, f"fold_{fold}_train_{out_tag}_pseudocells.h5ad")
        train_mapcsv  = os.path.join(out_dir, f"fold_{fold}_train_{out_tag}_mapping.csv")

        save_pseudocells_with_metadata(
            adata_orig=adata_train,                   # 원본 adata (전처리 전이어도 OK)
            labels=train_out['labels'],               # 길이 = adata_train.n_obs
            reps_dict=train_out['representatives'],   # cluster -> 벡터
            gene_names=train_out['hvg_mask'],         # 사용된 HVG 이름(Index)
            out_h5ad_path=train_outfile,
            mapping_csv_path=train_mapcsv,
            obs_cols_to_summarize=None                # 또는 ["manual_annotation", "patient", ...]
        )
        print(f"✅ saved TRAIN pseudo-cells -> {train_outfile}")
        print(f"🧭 saved CELL↔CLUSTER map  -> {train_mapcsv}")


        # --- TEST ---
        print("  ▶ TEST: build graph & cluster ...")
        test_out = run_fast_corr_pipeline(
            adata_test,
            n_hvg=1500,
            n_neighbors=30,
            resolution=1.0,
            rep_method='mean'
        )
        test_outfile = os.path.join(out_dir, f"fold_{fold}_test_{out_tag}_pseudocells.h5ad")
        save_pseudocells(test_out['representatives'], test_out['hvg_mask'], test_outfile)
        print(f"  ✅ saved TEST pseudo-cells  -> {test_outfile} (k={test_out['n_clusters']})")

        print(f"⏱ elapsed: {time.time()-t0:.1f}s")
