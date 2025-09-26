"""
[Single cell clustering based on cell-pair differentiability correlation and variance analysis](https://academic.oup.com/bioinformatics/article/34/21/3684/4996592)
논문 기반 코드 작성 (제공된 github 없음)

**Jiang et al., 2018 (Corr)**의 “cell-pair differentiability correlation” 아이디어를 Scanpy 기반으로 재현한 코드 스케치입니다.
- 원 논문은 U_ij를 “i, j 두 셀을 제외한 전체 대비 차등발현 패턴”으로 정의하고, 두 패턴 간 상관을 유사도로 씁니다(식 (1)–(4)) .
- 계산량을 줄이기 위해, 논문이 제안한 근사(O(np))처럼 U_ij 대신 셀 i 기준의 단일 Ui (전체 평균 대비 부호)로 바꾸고, 행(셀) 간 상관 ≈ differentiability correlation로 사용합니다(저자들이 명시한 근사) .
- 그 유사도(=상관)로 **계층적 군집(HCA)**을 하고, 논문 방식의 분산분해(ANOVA) 기반 K 결정을 단순화해 구현했습니다(섹션 2.2) .
- 이후 각 클러스터를 평균/합/메도이드로 대표하는 pseudo-cell을 만듭니다 (당신의 목적).
사용 라이브러리: scanpy, numpy, scipy, scikit-learn, pandas


무엇을 해주나요?
1. 각 adata에서 상위 HVG 선택 후 행렬 𝑋추출
2. Differentiability vector 𝑈 = sign(X - row_mean_excluded) (여기선 전체평균 근사) 생성
3. 행(셀) 간 상관(centered cosine)을 유사도로, 1 - similarity를 거리로 사용
4. HCA + 분산분해 기반 K 선택(첫 로컬 최대)
5. 대표 pseudo-cell 생성: mean/sum/medoid 옵션
6. 요청하신 5×5 반복 루프에 바로 끼워넣기


<< 사용 팁 / 옵션 >>
< 대표값 방식 >
- rep_method='mean': 각 클러스터 평균 → 가장 흔한 pseudo-cell
- rep_method='sum': 카운트 기반 downstream(DESeq2) 등과 친화적
- rep_method='medoid': 실제 셀 하나로 대표(해석 용이)
< K 결정 >
- 위 구현은 논문의 ANOVA 아이디어를 간소화했습니다(유전자 전반의 ∑𝑟𝑗 최대). 논문은 “여러 m(상위 DE gene 수) 조합 반복 후 최빈 K”로 안정화합니다 . 원리에 더 가깝게 하려면, choose_k_via_variance_analysis에서 여러 HVG 크기/부분집합을 샘플링해 최빈값을 채택하세요. 
< 계산량 >
- 본 근사 버전은 𝑈𝑖로 대체해 O(n·p) 생성 + 유사도 행렬 연산(n×n). n이 매우 크면 block 처리 또는 kNN sparsification 후 HCA 대신 그래프군집(Leiden)도 대안입니다.
< 일치성 >
- 원식 (4)의 ‘i, j 쌍 별 U_{ij}, U_{ji}’를 정확히 구현하려면 O(n²p)라 큰 데이터에서 부담이 큽니다. 논문도 n이 클 때 𝑈𝑖 근사를 제안합니다 .
"""

# pip install scanpy scipy scikit-learn pandas
# pip install -U pip setuptools wheel
# python -m pip install -U pip setuptools wheel # venv 안에서 pip 고정 업데이트
# python -m pip install "numpy==1.26.4"
# python -m pip install --no-cache-dir "scikit-misc==0.3.1"



# python data_cell_reduce2.py

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# ---------- Core: Differentiability correlation (approx. O(n p)) ----------
def differentiability_vectors(X):
    """
    X: (n_cells, n_genes) float array (normalized/log1p 권장)
    Return U: (n_cells, n_genes) in {-1,0,1}, where sign(X - global_mean)
    근사: 원 논문의 U_ij 대신 U_i (전체 평균 대비) 사용  
    """
    gmean = X.mean(axis=0, keepdims=True)  # ≈ (sum over l x_lk - x_ik) / (n-1)
    U = np.sign(X - gmean)  # -1, 0, +1
    return U

def rowwise_center_normalize(M):
    """
    각 행(셀)에서 평균 제거 후 L2 정규화 → 행 간 내적 = 피어슨 유사도 유사
    """
    M = M - M.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    return M / norms

def differentiability_correlation(X):
    """
    Return S: similarity (n x n) in [-1,1], and D: distance = 1 - S in [0,2]
    """
    U = differentiability_vectors(X)
    Uc = rowwise_center_normalize(U.astype(np.float32))
    S = Uc @ Uc.T  # (n x n) similarity
    np.fill_diagonal(S, 1.0)
    D = 1.0 - S
    return S, D

# ---------- ANOVA-based K selection (simplified from paper Section 2.2) ----------
def anova_R_score(X, labels):
    """
    섹션 2.2의 r_j = SSB_j / SST_j 를 여러 유전자에 대해 합함  
    X: (n, p); labels: (n,)
    """
    n, p = X.shape
    labs = np.asarray(labels)
    uniq = np.unique(labs)
    # 전체 평균
    mu = X.mean(axis=0, keepdims=True)
    # 유전자별 총제곱합(SST)
    SST = ((X - mu)**2).sum(axis=0) + 1e-12
    # 군집별 평균
    SSB = np.zeros(p, dtype=np.float64)
    for k in uniq:
        Xk = X[labs == k]
        nk = Xk.shape[0]
        if nk == 0: 
            continue
        muk = Xk.mean(axis=0, keepdims=True)
        SSB += nk * (muk - mu).ravel()**2
    r = SSB / SST  # 각 유전자별 비율
    return float(r.sum())

def choose_k_via_variance_analysis(X, D, k_range=range(2, 11), linkage='average'):
    """
    D: precomputed distance (n x n)
    X: expression matrix (n x p) — HVG 공간 사용 권장
    논문처럼 '첫 번째 로컬 최대'를 선택. 간단화를 위해 전역 최대로 대체 가능. 

    - 데이터 행렬 X 자체를 넣고 싶다 → linkage='ward', metric='euclidean' (HVG 공간에서 가장 흔히 쓰는 조합)
    - 사전 계산된 거리행렬 D를 쓰고 싶다 → linkage='average', metric='precomputed' (cosine similarity나 correlation distance 등을 직접 계산해서 넣을 때 유용)
    """
    n = D.shape[0]
    best_k, best_R, best_labels = None, -np.inf, None
    for k in k_range:
        model = AgglomerativeClustering(
            n_clusters=k, metric='euclidean', linkage=linkage
        )
        labels = model.fit_predict(D)
        R = anova_R_score(X, labels)
        if R > best_R:
            best_k, best_R, best_labels = k, R, labels
    return best_k, best_labels, best_R

# ---------- Cluster representatives ----------
def cluster_representatives(X, labels, method='mean'):
    """
    X: (n, p), labels: (n,), method in {'mean','sum','medoid'}
    Return dict: cluster_id -> 1D vector (p,)
    """
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
            # medoid: 클러스터 내부 평균거리 최소 셀
            from sklearn.metrics import pairwise_distances
            dk = pairwise_distances(Xk, Xk, metric='euclidean')
            ci = np.argmin(dk.mean(axis=1))
            reps[k] = Xk[ci]
        else:
            raise ValueError("method must be one of {'mean','sum','medoid'}")
    return reps

# ---------- End-to-end for one AnnData ----------
def run_corr_pipeline(adata, n_hvg=2000, rep_method='mean', k_range=range(2, 11)):
    """
    입력: AnnData (adata.X는 raw/normalized 모두 가능; 여기선 log1p 권장)
    출력: dict with labels, D(similarity-based distance), representatives
    """
    ad = adata.copy()
    # 간단 전처리 (필요시 조정)
    if 'log1p' not in ad.uns.get('pp', {}):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg, flavor='seurat_v3')
    ad = ad[:, ad.var['highly_variable']].copy()

    X = ad.X.A if hasattr(ad.X, 'A') else ad.X
    X = np.asarray(X, dtype=np.float32)

    # 선택적으로 gene-wise scaling (ANOVA 안정화에 도움)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # Corr 유사도 & 거리
    S, D = differentiability_correlation(X)  # D = 1 - S

    # K 선택 + 군집
    k_opt, labels, R = choose_k_via_variance_analysis(X, D, k_range=k_range)

    # 대표 pseudo-cell
    reps = cluster_representatives(X, labels, method=rep_method)

    out = dict(
        k_opt=k_opt,
        labels=labels,
        R_score=R,
        distance=D,
        similarity=S,
        representatives=reps,
        hvg_mask=np.asarray(ad.var['highly_variable'])
    )
    return out

# ---------- 5x5 루프 ----------
base = "/data/project/kim89/0804_data"
save = "/data/project/kim89/0804_data_cell_reduce"
results = {}  # (repeat, fold) -> outputs

for repeat in range(5):
    for fold in range(5):
        print(f"🔁 Repeat {repeat}, Fold {fold}")
        tr_path = os.path.join(base, f"repeat_{repeat}", f"fold_{fold}_train.h5ad")
        te_path = os.path.join(base, f"repeat_{repeat}", f"fold_{fold}_test.h5ad")

        adata_train = sc.read_h5ad(tr_path)
        adata_test  = sc.read_h5ad(te_path)

        # Train에 대해 Corr 기반 군집 & 대표 생성
        train_out = run_corr_pipeline(
            adata_train, n_hvg=2000, rep_method='mean', k_range=range(2, 16)
        )

        # Test에도 동일 파이프라인 적용(옵션: train의 HVG/스케일러 고정 가능)
        test_out = run_corr_pipeline(
            adata_test, n_hvg=2000, rep_method='mean', k_range=range(2, 16)
        )

        results[(repeat, fold)] = dict(train=train_out, test=test_out)

        # 예시: train 대표 pseudo-cell을 새로운 AnnData로 만들어 저장
        reps_mat = np.vstack([train_out['representatives'][k] for k in sorted(train_out['representatives'])])
        rep_ids  = [f"cluster_{k}" for k in sorted(train_out['representatives'])]
        var_hvg  = adata_train.var_names[train_out['hvg_mask']]
        ad_rep   = sc.AnnData(reps_mat, obs=pd.DataFrame(index=rep_ids), var=pd.DataFrame(index=var_hvg))
        out_dir  = os.path.join(save, f"repeat_{repeat}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"fold_{fold}_train_corr_pseudocells.h5ad")
        ad_rep.write(out_file)
        print(f"✅ saved pseudo-cells: {out_file}")
