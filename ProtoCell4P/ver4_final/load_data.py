import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder 

# (파이썬 3.7+ 이상) stdout을 줄단위 버퍼링으로 변경
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

class OurDataset(Dataset):
    def __init__(self, X, y, cell_id=None, gene_id=None, class_id=None, ct=None, ct_id=None):
        self.X = X
        self.y = y
        self.cell_id = cell_id
        self.gene_id = gene_id
        self.class_id = class_id
        self.ct = ct
        self.ct_id = ct_id
    # def __getitem__(self, i):
    #     if self.ct_id is not None:
    #         return self.X[i], self.y[i], self.ct[i]
    #     return self.X[i], self.y[i]
    # def __len__(self):
    #     return len(self.y)

def load_lupus(data_path = "../data/lupus/h5ad/CLUESImmVar_nonorm.V6.h5ad", task=None, load_ct=True, keep_sparse=True):
    # https://github.com/yelabucsf/lupus_1M_cells_clean
    assert task is not None
    
    adata = sc.read_h5ad(data_path)
    
    # before: (834096, 32738) | after: (834096, 24205)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.scale(adata, max_value=10, zero_center=True) # Unable to allocate 75.2 GiB

    if keep_sparse is False:
        adata.X = adata.X.toarray()

    print("Preprocessing Complete!")
    
    X = []
    y = []

    genes = adata.var_names.tolist()
    barcodes = adata.obs_names.tolist()
    cell_types = adata.obs["ct_cov"]

    ct_id = sorted(set(cell_types))
    mapping_ct = {c:idx for idx, c in enumerate(ct_id)}    
    ct = []

    for ind in tqdm.tqdm(sorted(set(adata.obs["ind_cov"]))):
        disease = list(set(adata.obs[adata.obs["ind_cov"] == ind]["disease_cov"]))
        pop = list(set(adata.obs[adata.obs["ind_cov"] == ind]["pop_cov"]))
        assert len(disease) == 1
        assert len(pop) == 1
        x = adata.X[adata.obs["ind_cov"] == ind]
        X.append(x)
        if task.lower() == "disease":
            y.append(disease[0])
        elif task.lower() == "population" or task.lower() == "pop":
            y.append(pop[0])
        ct.append([mapping_ct[c] for c in cell_types[adata.obs["ind_cov"] == ind]])

    class_id = sorted(set(y))
    mapping = {c:idx for idx, c in enumerate(class_id)}
    y = [mapping[c] for c in y]
    
    print(("[Size of dataset] "+" | ".join(["{:s}: {:d}"] * len(class_id))).format(*[item for i in range(len(class_id)) for item in [class_id[i], y.count(i)]]))

    # each sample in X has 4935 cells in average
    if load_ct:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
    else:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)

def load_cardio(data_dir = "../data/cardio", load_ct=True, keep_sparse=True):
    dat = sparse.load_npz(os.path.join(data_dir, "raw_counts.npz"))
    # genes = open(os.path.join(data_dir, "SCP1303expression614a0209771a5b0d7f033712DCM_HCM_Expression_Matrix_genes_V1.tsv")).read().strip().split("\n")
    genes = pd.read_csv(os.path.join(data_dir, "DCM_HCM_Expression_Matrix_genes_V1.tsv"), sep="\t", header=None).iloc[:,1].tolist()
    barcodes = open(os.path.join(data_dir, "DCM_HCM_Expression_Matrix_barcodes_V1.tsv")).read().strip().split("\n")
    meta = pd.read_csv(os.path.join(data_dir, "DCM_HCM_MetaData_V1.txt"), sep="\t").drop(axis=0,index=0).reset_index(drop=True)

    assert dat.shape[0] == len(barcodes) and len(barcodes) == meta.shape[0]
    assert dat.shape[1] == len(genes)
    
    cell_types = meta.cell_type__ontology_label
    ct_id = sorted(set(cell_types))
    mapping_ct = {c:idx for idx, c in enumerate(ct_id)}

    X = []
    y = []
    ct = []

    adata = sc.AnnData(dat.astype(np.float32), obs=barcodes, var=genes)

    # before: (592689, 36601) | after: (592689, 32151)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.scale(adata, max_value=10, zero_center=True)

    barcodes = adata.obs[0].tolist()
    genes = adata.var[0].tolist()
    
    if keep_sparse is False:
        adata.X = adata.X.toarray()

    for ind in tqdm.tqdm(sorted(set(meta.donor_id))):
        disease = list(set(meta.disease__ontology_label[meta.donor_id == ind]))
        assert len(disease) == 1
        x = adata.X[meta.donor_id == ind]
        X.append(x)
        y.append(disease[0])
        ct.append([mapping_ct[c] for c in cell_types[meta.donor_id == ind]])
    
    class_id = sorted(set(y))
    mapping = {c:idx for idx, c in enumerate(class_id)}
    y = [mapping[c] for c in y]
    
    # [Size of dataset] dilated cardiomyopathy: 11 | hypertrophic cardiomyopathy: 15 | normal: 16
    print(("[Size of dataset] "+" | ".join(["{:s}: {:d}"] * len(class_id))).format(*[item for i in range(len(class_id)) for item in [class_id[i], y.count(i)]]))
    
    # each sample in X has 14111 cells in average
    if load_ct:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
    else:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)    


def load_covid(data_dir = "../data/covid", load_ct=True, keep_sparse=True):
    dat = sparse.load_npz(os.path.join(data_dir, "RawCounts.npz"))
    genes = open(os.path.join(data_dir, "genes.txt")).read().strip().split("\n")
    barcodes = open(os.path.join(data_dir, "barcodes.txt")).read().strip().split("\n")
    meta = pd.read_csv(os.path.join(data_dir, "20210701_NasalSwab_MetaData.txt"), sep="\t").drop(axis=0,index=0).reset_index(drop=True)

    cell_types = pd.read_csv(os.path.join(data_dir, "20210220_NasalSwab_UMAP.txt"), sep="\t").drop(axis=0,index=0).reset_index(drop=True)["Category"]
    ct_id = sorted(set(cell_types))
    mapping_ct = {c:idx for idx, c in enumerate(ct_id)}

    X = []
    y = []
    ct = []

    adata = sc.AnnData(dat.astype(np.float32), obs=barcodes, var=genes)
    # adata = sc.AnnData(dat.astype(np.float32))
    # before: (32588, 32871) | after: (32588, 29696)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.scale(adata, max_value=10, zero_center=True)

    barcodes = adata.obs[0].tolist()
    genes = adata.var[0].tolist()
    
    if keep_sparse is False:
        adata.X = adata.X.toarray()

    for ind in tqdm.tqdm(sorted(set(meta.donor_id))):
        disease = list(set(meta.disease__ontology_label[meta.donor_id == ind]))
        assert len(disease) == 1
        if disease[0] == "long COVID-19" or disease[0] == "respiratory failure":
            continue
        x = adata.X[meta.donor_id == ind]
        X.append(x)
        y.append(disease[0])
        ct.append([mapping_ct[c] for c in cell_types[meta.donor_id == ind]])
    
    class_id = sorted(set(y))
    mapping = {c:idx for idx, c in enumerate(class_id)}
    y = [mapping[c] for c in y]
    
    # [Size of dataset] COVID-19: 35 | long COVID-19: 2 | normal: 15 | respiratory failure: 6
    print(("[Size of dataset] "+" | ".join(["{:s}: {:d}"] * len(class_id))).format(*[item for i in range(len(class_id)) for item in [class_id[i], y.count(i)]]))
    
    # each sample in X has 562 cells in average
    if load_ct:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
    else:
        return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)    

"""
### 단일 데이터 사용 시
import scanpy as sc
def load_icb_data(task, load_ct=False, keep_sparse=True):
    adata = sc.read_h5ad("../data_ours/icb/icb_adata_scAce.h5ad")

    # X: gene expression matrix
    if not keep_sparse:
        X = adata.X.toarray()
    else:
        X = adata.X

    print(f"Loaded X shape: {X.shape}")

    # Add normalization
    if hasattr(X, 'toarray'):
        X = X.toarray()
    # Zero-safe log transform
    X = np.log1p(X)
    X = X / (X.max() + 1e-8)
    
    print(f"Loaded X shape: {X.shape}")

    # y: label
    y = adata.obs["label"].astype("category").cat.codes.values

    # cell_id: cell barcodes
    barcodes = adata.obs.index.tolist()

    # gene_id: gene list
    genes = adata.var.index.tolist()

    # class_id: label categories
    class_id = adata.obs["label"].astype("category").cat.categories.tolist()
    print("Unique labels:", np.unique(y))

    print("Any NaN in X?", np.isnan(X).sum())
    print("Any NaN in Y?", np.isnan(y).sum())


    return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)


def load_covid_data(task, load_ct=False, keep_sparse=True):
    adata = sc.read_h5ad("../data_ours/covid/covid_adata_scAce.h5ad")

    # X: gene expression matrix
    if not keep_sparse:
        X = adata.X.toarray()
    else:
        X = adata.X

    print(f"Loaded X shape: {X.shape}")

    # Add normalization
    if hasattr(X, 'toarray'):
        X = X.toarray()
    # Zero-safe log transform
    X = np.log1p(X)
    X = X / (X.max() + 1e-8)
    
    print(f"Loaded X shape: {X.shape}")

    # y: label
    y = adata.obs["label"].astype("category").cat.codes.values

    # cell_id: cell barcodes
    barcodes = adata.obs.index.tolist()

    # gene_id: gene list
    genes = adata.var.index.tolist()

    # class_id: label categories
    class_id = adata.obs["label"].astype("category").cat.categories.tolist()
    print("Unique labels:", np.unique(y))

    print("Any NaN in X?", np.isnan(X).sum())
    print("Any NaN in Y?", np.isnan(y).sum())


    return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)

def load_cardio_data(task, load_ct=False, keep_sparse=True):
    adata = sc.read_h5ad("../data_ours/cardio/cardio_adata_scAce.h5ad")

    # X: gene expression matrix
    if not keep_sparse:
        X = adata.X.toarray()
    else:
        X = adata.X

    print(f"Loaded X shape: {X.shape}")

    # Add normalization
    if hasattr(X, 'toarray'):
        X = X.toarray()
    # Zero-safe log transform
    X = np.log1p(X)
    X = X / (X.max() + 1e-8)
    
    print(f"Loaded X shape: {X.shape}")

    # y: label
    y = adata.obs["label"].astype("category").cat.codes.values

    # cell_id: cell barcodes
    barcodes = adata.obs.index.tolist()

    # gene_id: gene list
    genes = adata.var.index.tolist()

    # class_id: label categories
    class_id = adata.obs["label"].astype("category").cat.categories.tolist()
    print("Unique labels:", np.unique(y))

    print("Any NaN in X?", np.isnan(X).sum())
    print("Any NaN in Y?", np.isnan(y).sum())


    return OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id)
"""
### 여기서부터는, split dataset 사용
def _resolve_paths(data_location, task, repeat, fold):
    # 1) data_location/{task}/repeat_x/fold_y_*.h5ad
    base1 = os.path.join(data_location, task)
    tr1 = os.path.join(base1, f"repeat_{repeat}", f"fold_{fold}_train.h5ad")
    te1 = os.path.join(base1, f"repeat_{repeat}", f"fold_{fold}_test.h5ad")
    if os.path.exists(tr1) and os.path.exists(te1):
        return tr1, te1

    # 2) data_location/repeat_x/fold_y_*.h5ad (task 없이)
    tr2 = os.path.join(data_location, f"repeat_{repeat}", f"fold_{fold}_train.h5ad")
    te2 = os.path.join(data_location, f"repeat_{repeat}", f"fold_{fold}_test.h5ad")
    if os.path.exists(tr2) and os.path.exists(te2):
        return tr2, te2

    raise FileNotFoundError(
        f"train/test h5ad not found under:\n 1) {tr1}\n 2) {tr2}\n"
        "경로를 확인해주세요."
    )

def _maybe_lognorm_dense(X, do_lognorm: bool):
    # do_lognorm=False면 그대로 반환
    if not do_lognorm:
        return X

    # sparse → dense
    if hasattr(X, "toarray"):
        X = X.toarray()
    # log1p + max 정규화
    X = np.log1p(X)
    maxv = X.max()
    if maxv > 0:
        X = X / maxv
    return X

def _pick_label_series(adata):
    if "disease__ontology_label" in adata.obs:
        return adata.obs["disease__ontology_label"]
    elif "label" in adata.obs:
        return adata.obs["label"]  # kindy는 'label' : {0,1,2}. ('disease_category' 하면 안됨. 4개임)
    else:
        raise KeyError("라벨 컬럼을 찾을 수 없습니다: 'disease__ontology_label' 또는 'disease_category' 필요")

def load_split_data(task, data_location, repeat, fold,
                    keep_sparse=False, lognorm=True, load_ct=False, seed=42, cell_type_annotation=None):
    train_path, test_path = _resolve_paths(data_location, task, repeat, fold)

    print(f"📂 Loading: {train_path}")
    adata_train = sc.read_h5ad(train_path)
    print(f"📂 Loading: {test_path}")
    adata_test = sc.read_h5ad(test_path)

    # === X ===
    X_train = adata_train.X if keep_sparse else (
        adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else adata_train.X
    )
    X_test = adata_test.X if keep_sparse else (
        adata_test.X.toarray() if hasattr(adata_test.X, "toarray") else adata_test.X
    )

    # === Optional LogNorm ===
    X_train = _maybe_lognorm_dense(X_train, do_lognorm=lognorm)
    X_test  = _maybe_lognorm_dense(X_test,  do_lognorm=lognorm)

    print(f"Loaded X train shape: {X_train.shape}")
    print(f"Loaded X test  shape: {X_test.shape}")

    # === y (일관 인코딩) ===
    ytr_raw = _pick_label_series(adata_train).astype(str).values
    yte_raw = _pick_label_series(adata_test).astype(str).values

    le = LabelEncoder()
    le.fit(np.concatenate([ytr_raw, yte_raw], axis=0))
    y_train = le.transform(ytr_raw)
    y_test  = le.transform(yte_raw)
    class_id = le.classes_.tolist()

    # === IDs ===
    barcodes_train = adata_train.obs.index.tolist()
    barcodes_test = adata_test.obs.index.tolist()
    genes = adata_train.var.index.tolist()  # 보통 train/test 같음

    # === ct ===
    # load_ct=True이고 cell_type_annotation이 주어졌을 때 CT를 인코딩해 넣는 부분
    ct_train = ct_test = ct_id = None
    if load_ct and cell_type_annotation is not None:
        if cell_type_annotation not in adata_train.obs or cell_type_annotation not in adata_test.obs:
            raise KeyError(f"{cell_type_annotation} not found in obs columns")
        ct_tr_raw = adata_train.obs[cell_type_annotation].astype(str).values
        ct_te_raw = adata_test.obs[cell_type_annotation].astype(str).values

        le_ct = LabelEncoder()
        le_ct.fit(np.concatenate([ct_tr_raw, ct_te_raw]))
        ct_id = le_ct.classes_.tolist()
        ct_train = [[c] for c in le_ct.transform(ct_tr_raw)]
        ct_test  = [[c] for c in le_ct.transform(ct_te_raw)]


    # NaN 체크
    def _nan_report(name, arr):
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        print(f"Any NaN in {name}? {np.isnan(arr).sum()}")

    _nan_report("X_train", X_train)
    _nan_report("X_test", X_test)
    print("Any NaN in y_train?", np.isnan(y_train).sum())
    print("Any NaN in y_test?", np.isnan(y_test).sum())
    print("train/test unique labels:", np.unique(y_train), np.unique(y_test))

    train_ds = OurDataset(
        X=X_train, y=y_train,
        cell_id=barcodes_train, gene_id=genes, class_id=class_id,
        ct=ct_train, ct_id=ct_id
    )
    test_ds  = OurDataset(
        X=X_test, y=y_test,
        cell_id=barcodes_test, gene_id=genes, class_id=class_id,
        ct=ct_test, ct_id=ct_id
    )
    return train_ds, test_ds