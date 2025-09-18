# run.py
import argparse
from config import Config
from utils import set_global_seed
import optuna
from utils import train_with_oom_retry
import numpy as np
import os

# (파이썬 3.7+ 이상) stdout을 줄단위 버퍼링으로 변경
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

HP_SPACE = {
    "lr":        [1e-4, 1e-3, 1e-2],
    "max_epoch": [50, 75, 100],
    "z_dim":     [32, 64, 128],  # encoder/decoder 출력 차원
    "h_dim":     [8, 16, 32],
    "n_proto":   [8, 16, 32],
    # pretrain epoch은 고정 {75}
}
# === run.py 상단에 추가 ===
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
# run.py
# run.py 상단
from types import SimpleNamespace


def make_trial_cfg(base_cfg, hp):
    tcfg = SimpleNamespace(
        # 튜닝 대상
        lr=hp["lr"],
        max_epoch=hp["max_epoch"],
        z_dim=hp["z_dim"],
        h_dim=hp["h_dim"],
        n_proto=hp["n_proto"],

        # 고정/복사
        n_layers=getattr(base_cfg, "n_layers", 2),
        batch_size=getattr(base_cfg, "batch_size", 4),
        device=getattr(base_cfg, "device", "cpu"),
        d_min=getattr(base_cfg, "d_min", 1),
        lr_pretrain=getattr(base_cfg, "lr_pretrain", 1e-2),
        max_epoch_pretrain=75,
        keep_sparse=getattr(base_cfg, "keep_sparse", True),
        test_step=getattr(base_cfg, "test_step", 1),

        # ★ 추가: lambdas, n_ct, seed까지 복사
        lambdas=getattr(base_cfg, "lambdas", None),
        n_ct=getattr(base_cfg, "n_ct", None),
        seed=getattr(base_cfg, "seed", None),

        # ★ 경로 필수 복사
        checkpoint_dir=getattr(base_cfg, "checkpoint_dir", None),
        log_dir=getattr(base_cfg, "log_dir", None),
    )
    # 필요시 디렉토리 보장
    if tcfg.checkpoint_dir:
        os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    if tcfg.log_dir:
        os.makedirs(tcfg.log_dir, exist_ok=True)
    return tcfg


def make_loaders_from_cfg(cfg, batch_size, phase):
    """
    cfg._build()로 만든 split을 그대로 사용해 DataLoader를 생성.
    phase: "valid_only" | "valid_plus_test"
    """
    train_loader = DataLoader(cfg.train_set, batch_size=batch_size if not cfg.subsample else 8*batch_size,
                              shuffle=True,  collate_fn=cfg.collate_fn)
    val_loader   = DataLoader(cfg.val_set,   batch_size=cfg.batch_size, shuffle=False, collate_fn=cfg.collate_fn)
    loaders = {"train": train_loader, "val": val_loader}
    if phase == "valid_plus_test":
        test_loader  = DataLoader(cfg.test_set,  batch_size=cfg.batch_size, shuffle=False, collate_fn=cfg.collate_fn)
        loaders["test"] = test_loader
    return loaders

def build_model_and_optim(tcfg, cfg, phase="valid_only"):
    """
    tcfg: trial용 하이퍼파라미터가 반영된 Config (z_dim/h_dim/n_proto/lr/max_epoch 등 변경)
    cfg : 데이터 split/메타를 가진 원본 Config (입력차원, 클래스수 등 추출)
    """
    # input_dim / n_classes / n_ct는 cfg._build()에서 이미 계산됨
    input_dim   = cfg.model.input_dim     # ProtoCell이 이미 한번 만들어져 있어 입력차원 보유
    n_classes   = cfg.n_classes
    n_ct        = cfg.n_ct
    lambdas     = cfg.lambdas
    # n_layers를 tcfg에서 읽되, 없으면 cfg의 값을 기본으로 사용
    n_layers = getattr(tcfg, "n_layers", getattr(cfg, "n_layers", 2))

    # fresh model 생성 (trial HP 반영)
    from model import ProtoCell, BaseModel
    if cfg.model_type == "ProtoCell":
        model = ProtoCell(input_dim, tcfg.h_dim, tcfg.z_dim, n_layers,
                          tcfg.n_proto, n_classes, lambdas, n_ct,
                          tcfg.device, d_min=tcfg.d_min)
    elif cfg.model_type == "BaseModel":
        model = BaseModel(input_dim, tcfg.h_dim, tcfg.z_dim, n_layers,
                          tcfg.n_proto, n_classes, lambdas, n_ct,
                          tcfg.device, d_min=tcfg.d_min)
    else:
        raise ValueError(f"Model {cfg.model_type} not supported")

    model.to(tcfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    loaders = make_loaders_from_cfg(cfg, batch_size=tcfg.batch_size, phase=phase)
    meta = dict(input_dim=input_dim, n_classes=n_classes, n_ct=n_ct)
    return model, optimizer, scheduler, loaders, meta

def run_pretrain(model, loaders, tcfg, logger=None, keep_sparse=True):
    """
    config.py의 pretrain 루프를 요약 이식. (구현은 기존과 동일 로직) 
    """
    model.train()
    optim_pre = torch.optim.Adam(model.parameters(), lr=tcfg.lr_pretrain)
    best_metric = None
    for epoch in range(tcfg.max_epoch_pretrain):
        train_loss = 0.0
        for bat in loaders["train"]:
            optim_pre.zero_grad()
            out = model.pretrain(*bat, sparse=keep_sparse)
            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            optim_pre.step()
            train_loss += loss.item() * len(bat[0])
        # 🔽 매 epoch 프린트
        print(f"[Pretrain] Epoch {epoch+1}/{tcfg.max_epoch_pretrain} "
              f"| Avg Loss={train_loss/len(loaders['train'].dataset):.4f}", flush=True)
        
        # 주기적 valid
        if (epoch + 1) % tcfg.test_step == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bat in loaders["val"]:
                    out = model.pretrain(*bat, sparse=keep_sparse)
                    loss = out[0] if isinstance(out, tuple) else out
                    val_loss += loss.item() * len(bat[0])
            curr_metric = - val_loss / len(loaders["val"].dataset)
            if best_metric is None or curr_metric > best_metric:
                ckpt_dir = getattr(tcfg, "checkpoint_dir", None)
                if ckpt_dir:                  # ← 없으면 그냥 저장 스킵
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pretrain.pt"))
                best_metric = curr_metric
            model.train()

def run_train_eval_only_valid(model, loaders, tcfg, logger=None, keep_sparse=True):

    """
    train + valid만으로 학습하고 valid AUC(또는 val loss 등)를 반환 → Optuna objective에서 사용.
    (기존 train() 루프 로직을 요약 이식. 지표 계산법은 config.py와 동일) 
    """
    print("train + valid만으로 학습하고 valid AUC를 반환 → Optuna objective에서 사용.")
    import time
    import torch
    from sklearn.metrics import roc_auc_score

    optim = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    pretrained = getattr(tcfg, "pretrained", False)

    best_metric = -1e18

    for epoch in range(tcfg.max_epoch):
        # =======================
        # Train
        # =======================
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for bat in loaders["train"]:
            optim.zero_grad()

            if len(bat) == 3:  # (x, y, ct)
                loss, logits, ct_logit = model(*bat, sparse=keep_sparse)
                # if not pretrained:
                #     loss = loss + model.pretrain(*bat, sparse=keep_sparse)[0]  # (loss, ct_logit)
            else:              # (x, y)
                loss, logits = model(*bat, sparse=keep_sparse)
                # if not pretrained:
                #     loss = loss + model.pretrain(*bat, sparse=keep_sparse)    # loss only
            
            loss.backward()
            optim.step()

            # 평균을 위해 샘플 수 기준 누적
            batch_n = len(bat[0])
            train_loss_sum += loss.item() * batch_n
            train_count += batch_n

        train_loss_avg = train_loss_sum / max(1, train_count)

            # optimizer는 바깥에서 받지 않았으므로 간단화를 위해 새로 선언
        # 간단화를 피하려면 build_model_and_optim에서 optimizer를 반환받아 여기서 사용하세요.
        # (아래 최종 코드 블록에서는 optimizer를 함께 사용)

        # valid
        # =======================
        # Validation
        # =======================
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        y_truth_list = []
        y_score_list = []

        with torch.no_grad():
            for bat in loaders["val"]:
                if len(bat) == 3:
                    loss, logits, ct_logit = model(*bat, sparse=keep_sparse)
                else:
                    loss, logits = model(*bat, sparse=keep_sparse)

                batch_n = len(bat[0])
                val_loss_sum += loss.item() * batch_n
                val_count += batch_n

                # AUC 계산을 위해서만 모음
                y_truth_list.append(bat[1].cpu())
                y_score_list.append(torch.softmax(logits, dim=1).cpu())

        val_loss_avg = val_loss_sum / max(1, val_count)

        # 메트릭 계산
        y_truth = torch.cat(y_truth_list)
        y_score = torch.cat(y_score_list)
        if y_score.shape[1] == 2:
            val_auc = roc_auc_score(y_truth.numpy(), y_score.numpy()[:, 1])
        else:
            val_auc = roc_auc_score(y_truth.numpy(), y_score.numpy(), multi_class="ovo")

        curr_metric = val_auc  # 필요시 val_loss를 조합해도 됨

        print(f"[epoch {epoch}] train_loss={train_loss_avg:.4f} "
            f"val_loss={val_loss_avg:.4f} val_auc={val_auc:.4f}", flush=True)

        if curr_metric > best_metric:
            best_metric = curr_metric

    return best_metric

def run_train_and_test(model, loaders, optimizer, tcfg, keep_sparse=True):
    """
    train + valid로 학습(베스트 모델) 후 test 평가를 반환
    반환: (auc, acc, cm, recall, precision) 형태
    (지표 계산은 config.py의 테스트 파트와 동일) 
    """

    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, recall_score, precision_score
    
    pretrained = getattr(tcfg, "pretrained", False)
    best_metric = -1e18
    best_state = None

    # === 학습 ===
    for epoch in range(tcfg.max_epoch):
        model.train()
        train_loss_sum, train_count = 0.0, 0

        for bat in loaders["train"]:
            optimizer.zero_grad()

            # out = model(*bat, sparse=keep_sparse)
            # loss, logits = out[0], out[1]                 # CT가 있어도 안전

            if len(bat) == 3:  # (x, y, ct)
                loss, logits, ct_logit = model(*bat, sparse=keep_sparse)
            else:              # (x, y)
                loss, logits = model(*bat, sparse=keep_sparse)
            
            if not pretrained:
                # pretrain() 반환: CT 있으면 (loss, ct_logit), 없으면 loss
                # 본 학습 중에도 pretrain 손실을 추가로 넣을지 여부를 결정하는 플래그예요.
                # tcfg.pretrained=True면 → 이미 별도로 pretrain 단계를 마친 상태라고 가정 → 본학습에서는 cross-entropy 분류 손실만 씀.
                # tcfg.pretrained=False면 → 사전학습을 안 했다고 가정 → 본학습 과정에서도 매 배치마다 pretrain 손실을 함께 추가하여 latent space를 같이 정돈.
                pre = model.pretrain(*bat, sparse=keep_sparse)
                loss = loss + (pre[0] if isinstance(pre, (list, tuple)) else pre)

            loss.backward()
            optimizer.step()

            batch_n = len(bat[0])
            train_loss_sum += loss.item() * batch_n
            train_count += batch_n

        avg_train_loss = train_loss_sum / max(1, train_count)

        # === 검증 ===
        model.eval()
        vloss_sum, vcount = 0.0, 0
        y_truth_list, y_score_list = [], []

        with torch.no_grad():
            for bat in loaders["val"]:
                out = model(*bat, sparse=keep_sparse)
                loss, logits = out[0], out[1]

                batch_n = len(bat[0])
                vloss_sum += loss.item() * batch_n
                vcount += batch_n

                y_truth_list.append(bat[1].cpu())
                y_score_list.append(torch.softmax(logits, dim=1).cpu())

        avg_val_loss = vloss_sum / max(1, vcount)

        y_truth = torch.cat(y_truth_list)
        y_score = torch.cat(y_score_list)
        y_pred = y_score.argmax(dim=1)

        if y_score.shape[1] == 2:
            val_auc = roc_auc_score(y_truth.numpy(), y_score.numpy()[:, 1])
        else:
            val_auc = roc_auc_score(y_truth.numpy(), y_score.numpy(), multi_class="ovo")
        val_acc = accuracy_score(y_truth.numpy(), y_pred.numpy())
        val_f1  = f1_score(y_truth.numpy(), y_pred.numpy(), average="macro")

        print(
            f"[Train+Valid] Epoch {epoch+1}/{tcfg.max_epoch} "
            f"| TrainLoss={avg_train_loss:.4f} "
            f"| ValLoss={avg_val_loss:.4f} "
            f"| ValAUC={val_auc:.4f} "
            f"| ValACC={val_acc:.4f} "
            f"| ValF1={val_f1:.4f}",
            flush=True
        )

        if val_auc > best_metric:
            from copy import deepcopy
            best_metric = val_auc
            best_state = deepcopy(model.state_dict())


    # === best 로드 후 test ===
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_truth_list, y_score_list = [], []
    with torch.no_grad():
        for bat in loaders["test"]:
            out = model(*bat, sparse=keep_sparse)
            logits = out[1]  # out[0]=loss, out[1]=logits
            y_truth_list.append(bat[1].cpu())
            y_score_list.append(torch.softmax(logits, dim=1).cpu())

    import torch as _torch
    y_truth = _torch.cat(y_truth_list)
    y_score = _torch.cat(y_score_list)
    y_pred = y_score.argmax(dim=1)

    acc = accuracy_score(y_truth.numpy(), y_pred.numpy())
    if y_score.shape[1] == 2:
        auc = roc_auc_score(y_truth.numpy(), y_score.numpy()[:, 1])
    else:
        auc = roc_auc_score(y_truth.numpy(), y_score.numpy(), multi_class="ovo")
    cm = confusion_matrix(y_truth.numpy(), y_pred.numpy())
    recall = recall_score(y_truth.numpy(), y_pred.numpy(), average="macro")
    precision = precision_score(y_truth.numpy(), y_pred.numpy(), average="macro")

    import numpy as _np
    cm_str = _np.array2string(cm, separator=' ', max_line_width=120)
    print(
        f"[Test] AUC={auc:.4f} | ACC={acc:.4f} | RECALL={recall:.4f} | PREC={precision:.4f}\n"
        f"[Test] Confusion Matrix (rows=true, cols=pred):\n{cm_str}",
        flush=True
    )

    return auc, acc, cm, recall, precision

def tune_one_fold(cfg):
    """
    cfg: 현재 fold의 데이터 split을 담은 Config (cfg._build() 완료 상태)
    내부에서 trial용 tcfg를 만들어 모형을 fresh하게 생성 → valid AUC 최대화를 목표로 탐색
    """
    def objective(trial):
        hp = suggest(trial)
        tcfg = make_trial_cfg(cfg, hp)    # ✅ 가벼운 설정 객체 생성
        tcfg.lr        = hp["lr"]
        tcfg.max_epoch = hp["max_epoch"]
        tcfg.z_dim     = hp["z_dim"]
        tcfg.h_dim     = hp["h_dim"]
        tcfg.n_proto   = hp["n_proto"]
        tcfg.max_epoch_pretrain = 75  # 고정

        def _call():
            model, optim, sched, loaders, meta = build_model_and_optim(tcfg, cfg, phase="valid_only")
            # pretrain
            # if tcfg.max_epoch_pretrain > 0:
            #     run_pretrain(model, loaders, tcfg, keep_sparse=cfg.keep_sparse)
            # train+valid만
            return run_train_eval_only_valid(model, loaders, tcfg, keep_sparse=cfg.keep_sparse)

        # OOM 백오프(8→4→2→1)
        return train_with_oom_retry(
            _call,
            backoff=(256, 128, 64, 32, 16, 8, 4, 2, 1),
            get_bs=lambda: tcfg.batch_size,
            set_bs=lambda b: setattr(tcfg, "batch_size", b),
        )

    study = optuna.create_study(direction="maximize")
    target = cfg.n_trials if hasattr(cfg, "n_trials") else 10
    attempts = 0
    while True:
        study.optimize(objective, n_trials=1, catch=())
        attempts += 1
        n_success = sum(t.state.name == "COMPLETE" for t in study.trials)
        if n_success >= target or attempts >= target * 50:
            break
    return sorted(study.best_trials, key=lambda t: t.value)[: getattr(cfg, "top_k", 1) ]

def suggest(trial):
    return {
        "lr": trial.suggest_categorical("lr", HP_SPACE["lr"]),
        "max_epoch": trial.suggest_categorical("max_epoch", HP_SPACE["max_epoch"]),
        "z_dim": trial.suggest_categorical("z_dim", HP_SPACE["z_dim"]),
        "h_dim": trial.suggest_categorical("h_dim", HP_SPACE["h_dim"]),
        "n_proto": trial.suggest_categorical("n_proto", HP_SPACE["n_proto"]),
    }

def main():
    p = argparse.ArgumentParser()

    # 필수/핵심
    p.add_argument("--data", default="split_datasets")
    p.add_argument("--data_location", required=True, help="base path of split datasets")
    p.add_argument("--tasks", nargs="+", default=["covid", "cardio", "kidney"])
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--n_folds", type=int, default=5)

    # 학습 하이퍼파라미터 (필요한 것만 예시로)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_epoch", type=int, default=75)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--h_dim", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_proto", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--exp_str", default=None) # 실험 이름(태그)”**이에요. 이 값이 Config 안에서 체크포인트/로그 경로를 만드는 데 꼭 쓰입니다.
    p.add_argument(
        "--cell_type_annotation",
        type=str,
        default=None,
        help="obs column name to use as cell type annotation (e.g. 'manual_annotation' or 'singler_annotation')"
)


    # 불리언 플래그는 BooleanOptionalAction 권장 (끄려면 --no-xxx)
    p.add_argument("--keep_sparse", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lognorm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--subsample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--load_ct", action=argparse.BooleanOptionalAction, default=False)

    # 기타
    p.add_argument("--model", default="ProtoCell", choices=["ProtoCell", "BaseModel"],
            help="which model architecture to use")
    p.add_argument("--d_min", type=float, default=1.0)
    p.add_argument("--lambda_1", type=float, default=1.0)
    p.add_argument("--lambda_2", type=float, default=1.0)
    p.add_argument("--lambda_3", type=float, default=1.0)
    p.add_argument("--lambda_4", type=float, default=1.0)
    p.add_argument("--lambda_5", type=float, default=1.0)
    p.add_argument("--lambda_6", type=float, default=1.0)
    p.add_argument("--lr_pretrain", type=float, default=1e-2)
    p.add_argument("--max_epoch_pretrain", type=int, default=0)
    
    # run.py - argparse 정의부 근처에 추가 (Python 3.8 호환)
    p.add_argument("--tune", dest="tune", action="store_true", help="enable Optuna tuning")
    p.add_argument("--no-tune", dest="tune", action="store_false", help="disable Optuna tuning")
    p.set_defaults(tune=True)

    p.add_argument("--n_trials", type=int, default=3, help="number of successful trials per fold")
    p.add_argument("--top_k", type=int, default=1, help="how many top trials to test on the test set")


    # 시드 (데이터 split엔 안 쓰고, 재현성용)
    p.add_argument("--base_seed", type=int, default=42)

    args = p.parse_args()

    print(">>> Start >>>")
    for task in args.tasks:
        # === [NEW] 이 task에서의 반복별 요약을 담아 둘 컨테이너
        task_repeat_summaries = []   # list of dicts: {"repeat": int, "auc": float, "acc": float, "recall": float, "prec": float}

        for repeat in range(args.n_repeats):
            # === [NEW] 한 repeat 안에서 5개 fold의 지표들을 모아두는 컨테이너
            fold_aucs, fold_accs, fold_recalls, fold_precs = [], [], [], []
            fold_cms = []  # 혼동행렬도 합산 가능

            for fold in range(args.n_folds):
                seed = args.base_seed + repeat * 10 + fold
                set_global_seed(seed)
                print(f"\n🔁 Task={task} | Repeat={repeat} | Fold={fold} | Seed={seed}")

                base = args.exp_str
                derived_exp = f"{base}_{task}_r{repeat}_f{fold}"
                
                print(">>> constructing Config", flush=True)

                cfg = Config(
                    data=args.data,
                    task=task,
                    data_location=args.data_location,
                    repeat=repeat,
                    fold=fold,
                    keep_sparse=args.keep_sparse,
                    lognorm=args.lognorm,
                    seed=seed,
                    # 아래는 기존 Config에 이미 있는 인자들 연결 (필요한 것만 예시)
                    model=args.model,   
                    cell_type_annotation=args.cell_type_annotation,   # ★ 추가
                    exp_str=derived_exp,
                    lr=args.lr,
                    max_epoch=args.max_epoch,
                    batch_size=args.batch_size,
                    h_dim=args.h_dim,
                    z_dim=args.z_dim,
                    n_layers=args.n_layers,
                    n_proto=args.n_proto,
                    device=args.device,
                    d_min=args.d_min,
                    lambda_1=args.lambda_1,
                    lambda_2=args.lambda_2,
                    lambda_3=args.lambda_3,
                    lambda_4=args.lambda_4,
                    lambda_5=args.lambda_5,
                    lambda_6=args.lambda_6,
                    pretrained=args.pretrained,
                    lr_pretrain=args.lr_pretrain,
                    max_epoch_pretrain=args.max_epoch_pretrain,
                    load_ct=args.load_ct,
                    eval=args.eval,
                    subsample=args.subsample,
                )
                # cfg 생성 직후
                cfg.tune = args.tune
                cfg.n_trials = args.n_trials
                cfg.top_k = args.top_k

                # === (A) 튜닝: valid만으로 HP 탐색 ===
                print(">>> TUNING", flush=True)
                best_trials = tune_one_fold(cfg) if getattr(cfg, "tune", True) else []

                # === (B) 같은 fold에서 최종 학습 + test 평가 ===
                print(">>> 최종 학습 + TEST 평가", flush=True)
                from copy import deepcopy
                auc = acc = rec = pre = None
                cm_use = None

                if best_trials:
                    best_fold = None
                    for t in best_trials:
                        hp = t.params
                        tcfg = make_trial_cfg(cfg, hp) 
                        tcfg.lr        = hp["lr"]
                        tcfg.max_epoch = hp["max_epoch"]
                        tcfg.z_dim     = hp["z_dim"]
                        tcfg.h_dim     = hp["h_dim"]
                        tcfg.n_proto   = hp["n_proto"]
                        tcfg.max_epoch_pretrain = 75

                        # ✅ 로그
                        logger = getattr(cfg, "logger", print)  # 안전하게

                        # ✅ 최종 하이퍼파라미터 로그 (tcfg 기준)
                        logger("*" * 40)
                        logger(f"[BEST from Optuna] value={t.value:.6f}  params={hp}")
                        logger(f"Learning rate: {tcfg.lr:.6f}")
                        logger(f"Max epoch: {int(tcfg.max_epoch)}")
                        logger(f"Batch size: {int(tcfg.batch_size)}")
                        logger(f"Test step: {int(tcfg.test_step)}")
                        logger(f"Hidden dim: {int(tcfg.h_dim)}")
                        logger(f"Latent dim (z_dim): {int(tcfg.z_dim)}")
                        logger(f"Number of prototypes: {int(tcfg.n_proto)}")
                        logger(f"Lambdas: {tcfg.lambdas}")
                        logger(f"D_min: {tcfg.d_min}")
                        logger(f"Device: {tcfg.device}")
                        logger(f"Seed: {tcfg.seed}")
                        logger("*" * 40)
                        logger("")

                        # 최종 학습 + 테스트
                        def _call():
                            model, optimizer, sched, loaders, meta = build_model_and_optim(tcfg, cfg, phase="valid_plus_test")
                            if getattr(tcfg, "max_epoch_pretrain", 0) > 0:
                                run_pretrain(model, loaders, tcfg, keep_sparse=cfg.keep_sparse)
                            return run_train_and_test(model, loaders, optimizer, tcfg, keep_sparse=cfg.keep_sparse)

                        try:
                            auc, acc, cm, rec, pre = train_with_oom_retry(
                                _call, backoff=(256, 128, 64, 32, 16, 8, 4, 2, 1),
                                get_bs=lambda: tcfg.batch_size,
                                set_bs=lambda b: setattr(tcfg, "batch_size", b),
                            )
                            if (best_fold is None) or (auc > best_fold["auc"]):
                                best_fold = dict(auc=auc, acc=acc, cm=cm, recall=rec, precision=pre)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"[WARN] OOM on final test for params {hp} → skip")
                            else:
                                raise

                    # 각 fold 결과 기록/로그 
                    if best_fold:
                        auc, acc, cm_use, rec, pre = (
                            best_fold["auc"], best_fold["acc"], best_fold["cm"], best_fold["recall"], best_fold["precision"]
                        )
                        print(f"[Fold{fold} Result] AUC={auc:.4f} | ACC={acc:.4f} | RECALL={rec:.4f} | PREC={pre:.4f}")
                    else:
                        # 전부 OOM 등으로 실패한 경우
                        print(f"[Fold{fold} Result] (no valid result)")

                else:
                    # 튜닝 없이 현재 cfg로
                    tcfg = deepcopy(cfg)
                    model, optimizer, sched, loaders, meta = build_model_and_optim(tcfg, cfg, phase="valid_plus_test")
                    if tcfg.max_epoch_pretrain > 0:
                        run_pretrain(model, loaders, tcfg, keep_sparse=cfg.keep_sparse)
                    auc, acc, cm_use, rec, pre = run_train_and_test(model, loaders, optimizer, tcfg, keep_sparse=cfg.keep_sparse)
                    print(f"[Fold{fold} Result] AUC={auc:.4f} | ACC={acc:.4f} | RECALL={rec:.4f} | PREC={pre:.4f}")

                # === [NEW] fold 결과를 컨테이너에 적재 (None이면 NaN 처리)
                def _tofloat(x): 
                    return float(x) if x is not None else np.nan

                fold_aucs.append(_tofloat(auc))
                fold_accs.append(_tofloat(acc))
                fold_recalls.append(_tofloat(rec))
                fold_precs.append(_tofloat(pre))
                if cm_use is not None:
                    fold_cms.append(np.asarray(cm_use, dtype=np.int64))



            # === [NEW] ⏸ 각 repeat 종료 후: 5개 fold 평균/표준편차 출력
            r_auc_mean, r_auc_std = np.nanmean(fold_aucs), np.nanstd(fold_aucs)
            r_acc_mean, r_acc_std = np.nanmean(fold_accs), np.nanstd(fold_accs)
            r_rec_mean, r_rec_std = np.nanmean(fold_recalls), np.nanstd(fold_recalls)
            r_pre_mean, r_pre_std = np.nanmean(fold_precs), np.nanstd(fold_precs)

            print(
                f"\n📌 [Repeat {repeat} Summary over {args.n_folds} folds]"
                f"  AUC={r_auc_mean:.4f}±{r_auc_std:.4f}"
                f" | ACC={r_acc_mean:.4f}±{r_acc_std:.4f}"
                f" | RECALL={r_rec_mean:.4f}±{r_rec_std:.4f}"
                f" | PREC={r_pre_mean:.4f}±{r_pre_std:.4f}"
            )

            # (선택) 혼동행렬 합산 표시
            if len(fold_cms) > 0:
                cm_sum = np.add.reduce(fold_cms)
                print("   Σ Confusion Matrix over folds:\n", cm_sum)

            # === [NEW] task 수준 요약에 추가(나중에 전체 요약 때 사용)
            task_repeat_summaries.append({
                "repeat": repeat,
                "auc": r_auc_mean,
                "acc": r_acc_mean,
                "recall": r_rec_mean,
                "prec": r_pre_mean,
            })


        # === [NEW] ✅ 모든 repeat 종료 후: 이 task에 대한 전체 요약 재정리 출력
        print("\n" + "="*72)
        print(f"🧾 Task='{task}' — Summary by Repeat (means over {args.n_folds} folds)")
        for rs in task_repeat_summaries:
            print(
                f"  - Repeat {rs['repeat']}: "
                f"AUC={rs['auc']:.4f} | ACC={rs['acc']:.4f} | RECALL={rs['recall']:.4f} | PREC={rs['prec']:.4f}"
            )

        # 최종(모든 repeat 평균) 1줄
        final_auc = np.nanmean([rs["auc"] for rs in task_repeat_summaries])
        final_acc = np.nanmean([rs["acc"] for rs in task_repeat_summaries])
        final_rec = np.nanmean([rs["recall"] for rs in task_repeat_summaries])
        final_pre = np.nanmean([rs["prec"] for rs in task_repeat_summaries])

        print("-"*72)
        print(
            f"✅ FINAL (Task='{task}', across {args.n_repeats} repeats, each averaged over {args.n_folds} folds)\n"
            f"   AUC={final_auc:.4f} | ACC={final_acc:.4f} | RECALL={final_rec:.4f} | PREC={final_pre:.4f}"
        )
        print("="*72 + "\n")

        
if __name__ == "__main__":
    main()

