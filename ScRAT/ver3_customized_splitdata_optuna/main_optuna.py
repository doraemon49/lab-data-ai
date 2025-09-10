from sklearn import metrics
from sklearn.metrics import accuracy_score
import scipy.stats as st
import torch
import torch.nn as nn

from torch.optim import Adam
from utils import *
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

from model_baseline import *
from Transformer import TransformerPredictor

from dataloader import *
from collections import defaultdict
import pandas as pd
import numpy as np

from collections import Counter

import functools, sys
# 모든 print() 기본 flush=True로 강제
print = functools.partial(print, flush=True)
# (파이썬 3.7+ 이상) stdout을 줄단위 버퍼링으로 변경
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def int_or_float(x):
    try:
        return int(x)
    except ValueError:
        return float(x)


parser = argparse.ArgumentParser(description='scRNA diagnosis')

parser.add_argument('--seed', type=int, default=240)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=3e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument("--task", type=str, default="severity")
parser.add_argument('--emb_dim', type=int, default=128)  # embedding dim
parser.add_argument('--h_dim', type=int, default=128)  # hidden dim of the model
parser.add_argument('--dropout', type=float, default=0.3)  # dropout
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument("--train_sample_cells", type=int, default=500,
                    help="number of cells in one sample in train dataset") # 학습 시 각 환자 샘플에서 500개 세포를 랜덤 선택 # 사용 1: 데이터 로드 시 환자 명수 조정. # 사용2: --all 1 해야지만 랜덤 샘플링 진행
parser.add_argument("--test_sample_cells", type=int, default=500,
                    help="number of cells in one sample in test dataset") # 테스트 시에도 동일하게 500개 세포 선택
parser.add_argument("--train_num_sample", type=int, default=20,
                    help="number of sampled data points in train dataset") # 한 명의 환자에서 500개의 세포를 20번 샘플링하여 20개의 bag 생성
parser.add_argument("--test_num_sample", type=int, default=100,
                    help="number of sampled data points in test dataset") # 테스트도 같은 방식으로 100개의 bag 생성
parser.add_argument('--model', type=str, default='Transformer')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--inter_only', type=_str2bool, default=False) # mixup된 샘플만 학습에 사용할지 여부
parser.add_argument('--same_pheno', type=int, default=0) # 같은 클래스끼리 mixup할지, 다른 클래스끼리 할지
# augment_num == 0 → mixup 안 함 → same_pheno 무의미
# augment_num > 0 → mixup 실행됨 → same_pheno가 환자 쌍 선택 규칙에 직접 영향
    # same_pheno=1 → 같은 클래스 내부의 다양성을 키우고 싶을 때 (클래스 간 경계를 흐리지 않음)
    # same_pheno=-1 → 클래스 간 경계를 부드럽게 하여 모델 일반화 유도
    # same_pheno=0 → 데이터 수가 적거나 클래스 불균형이 심할 때 무작위로 섞어 다양성 극대화
parser.add_argument('--augment_num', type=int, default=100) # Mixup된 새로운 가짜 샘플을 몇 개 생성할지
parser.add_argument('--alpha', type=float, default=1.0) # mixup의 비율 (Beta 분포 파라미터)
parser.add_argument('--repeat', type=int, default=5)
parser.add_argument('--all', type=int, default=1)
# all == 0:
    # sample_cells 만큼 랜덤 샘플링 (np.random.choice)
# all == 1
    # 샘플링을 건너뛰고 '해당 환자(혹은 라벨)의 모든 셀'을 그대로 사용
parser.add_argument('--min_size', type=int, default=6000)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--pca', type=_str2bool, default=False)
parser.add_argument('--mix_type', type=int, default=1)
parser.add_argument('--norm_first', type=_str2bool, default=False)
parser.add_argument('--warmup', type=_str2bool, default=False)
parser.add_argument('--top_k', type=int, default=1)

parser.add_argument('--cell_type_annotation', type=str, default="manual_annotation",
    help="사용할 cell type annotation 컬럼명 (manual_annotation 또는 singler_annotation)")

args = parser.parse_args()

# print("# of GPUs is", torch.cuda.device_count())
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

patient_summary = {}
stats = {}
stats_id = {}

if args.task == 'haniffa' or args.task == 'combat':
    label_dict = {0: 'Non Covid', 1: 'Covid'}
elif args.task == 'severity':
    label_dict = {0: 'mild', 1: 'severe'}
elif args.task == 'stage':
    label_dict = {0: 'convalescence', 1: 'progression'}
elif args.task == 'custom_cardio':
    label_dict = {
    0: 'normal',
    1: 'hypertrophic cardiomyopathy',
    2: 'dilated cardiomyopathy'
}
elif args.task == 'custom_covid':
    label_dict = {0: 'normal', 1: 'COVID-19'}
elif args.task == 'custom_kidney':
    label_dict = {
        0:'Healthy',  # 두 healthy subtype을 하나의 대표명으로
        1:'CKD',
        2:'AKI'
} 

# def safe_gather(data_mat, index_tensor):
#     # index_tensor: LongTensor (B, N) 또는 (N,)
#     idx = index_tensor.clone()
#     idx[idx < 0] = 0  # -1 패딩은 0으로 치환해 안전 인덱싱
#     return torch.from_numpy(data_mat[idx.cpu().numpy()])


# main.py - import 근처
import optuna

# 후보 셋
HP_SEARCH_SPACE = {
    "learning_rate": [1e-4, 1e-3, 1e-2],
    "epochs": [100],  # 고정
    "heads": [1, 2, 4],
    "dropout": [0.0, 0.3, 0.5, 0.7],
    "weight_decay": [1e-4, 1e-3, 1e-2],
    # "do_aug": [True, False],
    "emb_dim": [8, 32, 64],
    "augment_num": [100],   # (질문 명세상 고정)
    "pca": [False],         # (질문 명세상 고정)
}

def suggest_hparams(trial):
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", HP_SEARCH_SPACE["learning_rate"]),
        "epochs": trial.suggest_categorical("epochs", HP_SEARCH_SPACE["epochs"]),
        "heads": trial.suggest_categorical("heads", HP_SEARCH_SPACE["heads"]),
        "dropout": trial.suggest_categorical("dropout", HP_SEARCH_SPACE["dropout"]),
        "weight_decay": trial.suggest_categorical("weight_decay", HP_SEARCH_SPACE["weight_decay"]),
        # "do_aug": trial.suggest_categorical("do_aug", HP_SEARCH_SPACE["do_aug"]),
        "emb_dim": trial.suggest_categorical("emb_dim", HP_SEARCH_SPACE["emb_dim"]),
        "augment_num": trial.suggest_categorical("augment_num", HP_SEARCH_SPACE["augment_num"]),
        "pca": trial.suggest_categorical("pca", HP_SEARCH_SPACE["pca"]),
    }


# === OOM 최후 안전장치 ===
# === OOM 최후 안전장치: 8 -> 4 -> 2 -> 1 단계적 재시도 ===
def train_with_oom_retry(call, backoff=(8, 4, 2, 1)):
    """
    call: 인자 없는 함수(람다)로 train(...) 호출을 감쌉니다.
    backoff: OOM 발생 시 순차적으로 시도할 batch_size 후보들.
             각 후보는 '원래 batch_size'와의 min()으로 적용됩니다.
    """
    orig_bs = getattr(args, "batch_size", None)
    last_err = None
    try:
        # 1) 원래 배치로 1차 시도
        try:
            return call()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            last_err = e
            print("CUDA OOM 발생 — 단계적 백오프 시작.")

        # 2) 단계적 백오프: 8 → 4 → 2 → 1
        for bs in backoff:
            # 캐시 비우기(가능하면)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if orig_bs is not None:
                new_bs = min(orig_bs, bs)
                print(f"[OOM RETRY] batch_size {orig_bs} → {new_bs} 낮춰서 다시 시도합니다.")
                args.batch_size = new_bs

            try:
                return call()   # 성공 시 바로 반환
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    last_err = e
                    print(f"[OOM RETRY] batch_size={args.batch_size}에서도 OOM. batch_size를 더 낮춰서 다시 시도합니다... (batch_size 1이었을 경우 중단합니다.)")
                    continue
                # OOM 외 에러는 그대로 전파
                raise

        # 3) 모두 실패 → 마지막 OOM 전파
        raise last_err

    finally:
        # ✅ 여기서 항상 원복 (성공/실패/return/raise 모두 포함)
        if orig_bs is not None:
            args.batch_size = orig_bs




def train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test,
          data_augmented, data, eval_test: bool = True):    
    dataset_1 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='train')
    dataset_2 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='test')
    dataset_3 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='val')
    train_loader = torch.utils.data.DataLoader(dataset_1, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=dataset_1.collate)
    test_loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False, collate_fn=dataset_2.collate)
    valid_loader = torch.utils.data.DataLoader(dataset_3, batch_size=1, shuffle=False, collate_fn=dataset_3.collate)


    num_train_samples = len(dataset_1)   # ← 실제 train 샘플(환자/백) 개수
    print(f"👉 train samples(학습 샘플(bag))={num_train_samples}, batch_size={args.batch_size} -> steps(샘플수/batch크기)={len(train_loader)}")
    print(f"👉 valid samples={len(dataset_3)}, batch_size=1 -> steps={len(valid_loader)}")
    print(f"👉 test  samples={len(dataset_2)}, batch_size=1 -> steps={len(test_loader)}")
    # steps(len(~_loader)) 설명 :
    # train은 mixup+sampling으로 bag이 늘어난 것(covid : 126bags).
    # test는 mixup 안 쓰고 --all 1이라 환자=bag(14)로 고정.
    # test bag을 늘리려면 --all 0로 전환하고 --test_num_sample을 키우면 됩니다.


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = data_augmented[0].shape[-1]

    output_class = len(set(labels_))
    if (output_class == 2):
        output_class = 1

    if args.task in ('custom_cardio', 'custom_kidney'):
        output_class = 3
    elif args.task == 'custom_covid' or output_class == 2:
        output_class = 1
        print("output_class : ", output_class)

    if args.model == 'Transformer':
        model = TransformerPredictor(input_dim=input_dim, model_dim=args.emb_dim, num_classes=output_class,
                                     num_heads=args.heads, num_layers=args.layers, dropout=args.dropout,
                                     input_dropout=0, pca=args.pca, norm_first=args.norm_first)
    elif args.model == 'feedforward':
        model = FeedForward(input_dim=input_dim, h_dim=args.emb_dim, cl=output_class, dropout=args.dropout)
    elif args.model == 'linear':
        model = Linear_Classfier(input_dim=input_dim, cl=output_class)
    elif args.model == 'scfeed':
        model = scFeedForward(input_dim=input_dim, cl=output_class, model_dim=args.emb_dim, dropout=args.dropout, pca=args.pca)

    model = nn.DataParallel(model)
    torch.cuda.empty_cache() # 모델을 GPU로 올리기 전에 캐시 비우기
    model.to(device)
    best_model = model

    allocated = torch.cuda.memory_allocated()
    # 현재 예약된 메모리 (캐시 포함)
    reserved = torch.cuda.memory_reserved()

    # MB 단위로 보기
    print(f"Memory Allocated: {allocated / 1024 ** 2:.2f} MB")
    print(f"Memory Reserved: {reserved / 1024 ** 2:.2f} MB")

    print(device)

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.warmup:
        import transformers
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.epochs // 10,
                                                                 num_training_steps=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)
    sigmoid = torch.nn.Sigmoid().to(device)

    max_acc, max_epoch, max_auc, max_loss, max_valid_acc, max_valid_auc = 0, 0, 0, 0, 0, 0
    test_accs, valid_aucs, train_losses, valid_losses, train_accs, test_aucs = [], [0.], [], [], [], []
    best_valid_loss = float("inf")
    wrongs = []
    trigger_times = 0
    patience = 2
    for ep in (range(1, args.epochs + 1)):
        model.train()
        train_loss = []


        for batch in (train_loader):
            x_ = torch.from_numpy(data_augmented[batch[0]]).float().to(device)
            y_ = batch[1].to(device)
            mask_ = batch[3].to(device)
            optimizer.zero_grad()

            out = model(x_, mask_)

            if output_class==1:
                loss = nn.BCELoss()(sigmoid(out), y_)
            elif output_class==3:
                # === Multi-class (예: 3-class) 학습 손실 ===
                # out: (B, N, C) 또는 (N, C) 로 가정. 마지막 차원이 C(클래스 수).
                # y_: (B, 1) 형태(float)로 들어오므로 long으로 변환 후, 셀 수(N)만큼 확장.
                if out.dim() == 3:
                    B, N, C = out.shape
                    targets = y_.long().view(B, 1).expand(B, N).long()           # (B, N)
                    loss = nn.CrossEntropyLoss()(out.reshape(-1, C), targets.reshape(-1))
                elif out.dim() == 2:
                    N, C = out.shape
                    targets = y_.view(-1).long()           # ✅ [B]
                    # 필요하면 out의 배치 B와 일치하는지 assert
                    assert targets.shape[0] == out.shape[0], f"{targets.shape} vs {out.shape}"

                    loss = nn.CrossEntropyLoss()(out, targets)
                else:
                    raise RuntimeError(f"Unexpected logits shape for multiclass: {out.shape}")


            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

        scheduler.step()

        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)

        if ep % 1 == 0:
            valid_loss = []
            model.eval()
            pred = []
            true = []
            with torch.no_grad():
                for batch in (valid_loader):
                    x_ = torch.from_numpy(data[batch[0]]).float().to(device).squeeze(0)
                    y_ = batch[1].int().to(device)

                    out = model(x_)
                    if output_class == 1:
                        out = sigmoid(out)

                        loss = nn.BCELoss()(out, y_ * torch.ones(out.shape).to(device))
                        valid_loss.append(loss.item())

                        out = out.detach().cpu().numpy()

                        # majority voting
                        f = lambda x: 1 if x > 0.5 else 0
                        func = np.vectorize(f)
                        out = np.argmax(np.bincount(func(out).reshape(-1))).reshape(-1)
                        pred.append(out)
                        y_ = y_.detach().cpu().numpy()
                        true.append(y_)
                    else:
                        # === Multiclass ===
                        # 셀별 로짓이 나오면 bag 평균으로 축약 후 CE
                        if out.dim() == 3:          # [B=1, N, C]
                            logits = out.mean(dim=1)        # [1, C]
                        else:                        # [1, C] 또는 [N, C]인 경우
                            if out.dim() == 2 and out.shape[0] > 1:  # [N, C]라면 N축 평균으로 bag 로짓 만들기
                                logits = out.mean(dim=0, keepdim=True)  # [1, C]
                            else:
                                logits = out                         # [1, C]

                        loss = nn.CrossEntropyLoss()(logits, y_.view(-1).long())
                        valid_loss.append(loss.item())

                        # 예측/정답 기록
                        probs = torch.softmax(logits, dim=-1)     # [1, C]
                        pred_cls = int(torch.argmax(probs, dim=-1).item())
                        pred.append(pred_cls)
                        true.append(int(y_.view(-1).item()))



            # pred = np.concatenate(pred)
            # true = np.concatenate(true)

            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_losses.append(valid_loss)

            if (valid_loss < best_valid_loss):
                best_model = copy.deepcopy(model)
                max_epoch = ep
                best_valid_loss = valid_loss
                max_loss = train_loss

            print("Epoch %d, Train Loss %f, Valid_loss %f" % (ep, train_loss, valid_loss))

            # Early stop
            if (ep > args.epochs - 50) and ep > 1 and (valid_loss > valid_losses[-2]):
                trigger_times += 1
                if trigger_times >= patience:
                    break
            else:
                trigger_times = 0

    if not eval_test:
        # 튜닝 목적: test는 건드리지 않고 valid 기준만 반환
        return None, None, None, None, None, best_valid_loss
    
    # ---- 이하 기존 test 평가 코드 유지 ----
    best_model.eval()
    test_id = []
    wrong = []
    if output_class == 1:
        pred = []
        true = []
        prob = []

    else:
        preds_mc,  trues_mc,  probvecs_mc = [], [], []   # probvecs_mc: 각 샘플의 softmax 확률 벡터

    with torch.no_grad():
        for batch in (test_loader):
            x_ = torch.from_numpy(data[batch[0]]).float().to(device).squeeze(0)
            y_ = batch[1].int().numpy()
            id_ = batch[2][0] # 환자/샘플 ID

            out = best_model(x_)

            if output_class == 1:
            # === Binary Classification ===
                out = sigmoid(out)
                out = out.detach().cpu().numpy().reshape(-1)

                y_ = y_[0][0]
                true.append(y_)

                if args.model != 'Transformer':
                    prob.append(out[0])
                else:
                    prob.append(out.mean())

                # majority voting
                f = lambda x: 1 if x > 0.5 else 0
                func = np.vectorize(f)
                out = np.argmax(np.bincount(func(out).reshape(-1))).reshape(-1)[0]
                pred.append(out)
                test_id.append([batch[2][0]])
                if out != y_:
                    wrong.append([batch[2][0]])

                # 출력
                # print(f"-- true: {y_}, pred: {out}")
                
                # id_는 list, 보통 batch_size=1이라 id_[0]만 있음
                cell_indices = np.array(id_[0])  # [163460, 163461, ...]
                first_idx = int(cell_indices[0]) # 전역 셀 인덱스 하나
                pid = patient_id_all[first_idx]      # ✅ 실제 환자 ID 문자열

                print(f"환자ID={pid} -- true: {y_} -- pred: {out}")


            else:
                # ===== Multiclass =====                
                # y_가 numpy일 수도, torch일 수도 있을 때의 범용 처리
                if torch.is_tensor(y_):
                    y_true = int(y_.detach().cpu().reshape(-1)[0].item())
                else:
                    y_true = int(np.array(y_).reshape(-1)[0])


                # logits -> bag 로짓
                if out.dim() == 3:       # [B=1, N, C]
                    logits = out.mean(dim=1)                 # [1, C]
                elif out.dim() == 2:     # [N, C] 또는 [1, C]
                    logits = out.mean(dim=0, keepdim=True) if out.shape[0] > 1 else out   # [1, C]
                else:
                    raise RuntimeError(f"Unexpected logits shape: {out.shape}")

                p = torch.softmax(logits, dim=-1).squeeze(0)  # [C]
                y_pred = int(torch.argmax(p).item())

                probvecs_mc.append(p.cpu().numpy())  # AUC용 확률 벡터
                preds_mc.append(y_pred)
                trues_mc.append(y_true)
                test_id.append(batch[2][0])         # or test_patient_id[...] 로깅
                if y_pred != y_true:
                    wrong.append(test_id[-1])

                # print(f"-- true: {y_true}, pred: {y_pred}")
                
                # id_는 list, 보통 batch_size=1이라 id_[0]만 있음
                cell_indices = np.array(id_[0])  # [163460, 163461, ...]
                first_idx = int(cell_indices[0]) # 전역 셀 인덱스 하나
                pid = patient_id_all[first_idx]      # ✅ 실제 환자 ID 문자열

                print(f"환자ID={pid} -- true: {y_} -- pred: {out}")


    # if len(wrongs) == 0:
    #     wrongs = set(wrong)
    # else:
    #     wrongs = wrongs.intersection(set(wrong))
    
    # ====== 집계 및 지표 ======
    if output_class == 1:
        test_acc = accuracy_score(true, pred)
        try:
            test_auc = metrics.roc_auc_score(true, prob)
        except ValueError:
            print("⚠️ AUC 계산 불가: test set 클래스가 단일입니다.")
            test_auc = np.nan

        cm = confusion_matrix(true, pred).ravel()
        if len(cm) == 4:
            recall    = cm[3] / (cm[3] + cm[2]) if (cm[3] + cm[2]) > 0 else np.nan
            precision = cm[3] / (cm[3] + cm[1]) if (cm[3] + cm[1]) > 0 else np.nan
        else:
            print("⚠️ Skipping evaluation due to insufficient class diversity")
            recall = precision = np.nan

        # 로그
        # for i in range(len(pred)):
        #     print(f"{test_ids[i]} -- true: {label_dict[true[i]]} -- pred: {label_dict[pred[i]]}")

    else: # multi-class
        test_acc = accuracy_score(trues_mc, preds_mc)
        try:
            # probvecs_mc: (num_samples, C) 로 변환
            prob_matrix = np.vstack(probvecs_mc)   # shape [N, C]
            test_auc = metrics.roc_auc_score(trues_mc, prob_matrix, multi_class='ovo') # , average='weighted'
        except ValueError:
            print("⚠️ AUC 계산 불가: test set에 모든 클래스가 포함되지 않음")
            test_auc = np.nan

        cm = confusion_matrix(trues_mc, preds_mc)
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, _, _ = precision_recall_fscore_support(
            trues_mc, preds_mc, average='macro', zero_division=0
        )

    # 공통 출력
    print("Best performance: Epoch %d, Loss %f, Test ACC %f, Test AUC %f, Test Recall %f, Test Precision %f" %
        (max_epoch, max_loss, test_acc, test_auc, recall, precision))
    print("Confusion Matrix:\n", cm)
    
    # train() 마지막 반환 직전
    # 매 fold/trial마다 학습이 끝날 때 GPU 캐시를 비워 줍니다.
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # train() 맨 끝
    return test_auc, test_acc, cm, recall, precision, best_valid_loss


    # # AUC 계산 방식 분기
    # try:
    #     if output_class == 1:
    #         test_acc = accuracy_score(true, pred)
    #         test_auc = metrics.roc_auc_score(true, prob)
    #     else:
    #         test_acc = accuracy_score(trues_mc, preds_mc)
    #         # probvecs_mc: (num_samples, C) 로 변환
    #         prob_matrix = np.vstack(probvecs_mc)   # shape [N, C]
    #         test_auc = metrics.roc_auc_score(trues_mc, prob_matrix, multi_class='ovr', average='weighted')
    # except ValueError:
    #     print("⚠️ AUC 계산 불가: test set에 모든 클래스가 존재하지 않음 (이진분류의 경우 하나의 class만 존재. 삼중분류의 겨우 두개의 class만 존재)")
    #     test_auc = np.nan


    # # for idx in range(len(pred)):
    # #     print(f"{test_id[idx]} -- true: {label_dict[true[idx]]} -- pred: {label_dict[pred[idx]]}")
    # test_accs.append(test_acc)

    # print("true : ", true)
    # print("pred : ", pred)

    # # Confusion Matrix 및 지표 분기 # 하나의 임계값(보통 0.5)에서의 스냅샷일 뿐
    # if output_class == 1:
    #     cm = confusion_matrix(true, pred).ravel()

    #     if len(cm) == 4:
    #         recall = cm[3] / (cm[3] + cm[2]) if (cm[3] + cm[2]) > 0 else np.nan
    #         precision = cm[3] / (cm[3] + cm[1]) if (cm[3] + cm[1]) > 0 else np.nan
    #     else:
    #         print("⚠️ Skipping evaluation due to insufficient class diversity")
    #         recall = precision = np.nan
    #     print("Confusion Matrix: " + str(cm))

    # else:  #  multiclass 일 때의 cm, recall, precision 정의해주세요
    #     # 멀티클래스 혼동행렬 (정방 행렬)
    #     cm = confusion_matrix(trues_mc, preds_mc)
    #     # 매크로 평균(클래스 균등 가중) — 상황에 따라 'weighted'를 써도 됩니다.
    #     from sklearn.metrics import precision_recall_fscore_support
    #     precision, recall, _, _ = precision_recall_fscore_support(
    #         trues_mc, preds_mc, average='macro', zero_division=0
    #     )
    #     print("Confusion Matrix:\n", cm)

    # # 로그
    # for i in range(len(preds)):
    #     print(f"{test_ids[i]} -- true: {label_dict[trues[i]]} -- pred: {label_dict[preds[i]]}")
    # print("true : ", trues)
    # print("pred : ", preds)
    # if output_class == 1:
    #     print("Confusion Matrix:", cm)
    # else:
    #     print("Confusion Matrix:\n", cm)


    # print("Best performance: Epoch %d, Loss %f, Test ACC %f, Test AUC %f, Test Recall %f, Test Precision %f" % (
    # max_epoch, max_loss, test_acc, test_auc, recall, precision))
    # print("Confusion Matrix: " + str(cm))
    # for w in wrongs:
    #     v = patient_summary.get(w, 0)
    #     patient_summary[w] = v + 1

    # return test_auc, test_acc, cm, recall, precision


if args.model != 'Transformer':
    args.repeat = 60

# 이미 만들어 둔 train/valid 분할과, mixup/sampling 결과를 받아 **valid 성능(=best_valid_loss)**를 최소화하는 방향으로 튜닝합니다.
# 여기서 test는 건드리지 않습니다.
def optuna_tune_one_fold(prep_A, n_trials=5, top_k=args.top_k):
    def objective(trial):
        hp = suggest_hparams(trial)
        # 하이퍼 주입
        args.learning_rate = hp["learning_rate"]
        args.weight_decay  = hp["weight_decay"]
        args.epochs        = hp["epochs"]
        args.heads         = hp["heads"]
        args.dropout       = hp["dropout"]
        args.emb_dim       = hp["emb_dim"]
        args.augment_num   = hp["augment_num"]
        args.pca           = hp["pca"]

        # 위험 조합 사전 가드 (예: 메모리 빡센 조합 최소 완화)
        orig_bs = args.batch_size
        if args.heads == 4 and args.emb_dim >= 64:
            args.batch_size = min(args.batch_size, 8) ## 기본값 16에서, min(현재값 or 8)로 낮춤

        try:
            # 튜닝은 valid만 (eval_test=False)
            def _call():
                data_for_training = prep_A["data_augmented"]  # 현재 설계상 튜닝은 mixup 사용
                return train(
                    prep_A["x_train"], prep_A["x_valid"], [],
                    prep_A["y_train"], prep_A["y_valid"], np.empty((0,1)),
                    prep_A["id_train"], [],
                    data_for_training, prep_A["data"], eval_test=False
                )
            try:
                # ✅ 다단계 OOM 백오프으로 감싸기
                _, _, _, _, _, best_vloss = train_with_oom_retry(_call, backoff=(8, 4, 2, 1))
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    trial.set_user_attr("oom", True)
                    import optuna
                    # ✅ 이 트라이얼은 '실패'가 아니라 'Pruned'로 종료 → 성공 trial 아님
                    raise optuna.exceptions.TrialPruned("Pruned due to OOM after backoff")
                raise
        finally:
            # 배치 사이즈 원복
            args.batch_size = orig_bs

        return best_vloss # 검증(valid)에서 얻은 손실 중 가장 낮은 값(best_valid_loss) 을 돌려줌

    study = optuna.create_study(direction="minimize") # valid loss를 최소화하는 방향으로 최적의 하이퍼파라미터를 탐색
                                                    # search 알고리즘: 기본은 TPE (Tree-structured Parzen Estimator). 즉, **완전 무작위(random search)**가 아니라, 앞 trial들의 성능 분포를 참고해서 다음 trial을 점점 더 promising한 영역에서 샘플링합니다.

    # ✅ ‘성공(Complete) trial’을 정확히 n_trials 개 모을 때까지 반복
    target_success = n_trials
    max_attempts   = n_trials * 50  # 무한 루프 방지 상한 (상황 따라 조정)
    attempts = 0
    while True:
        # 1개씩 시도 (OOM/Pruned면 성공 카운트에 미포함)
        study.optimize(objective, n_trials=1, catch=())
        attempts += 1

        # 현재 성공(=COMPLETE) trial 수 집계
        n_success = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        if n_success >= target_success:
            break
        if attempts >= max_attempts: # 모든 횟수 소진..
            print(f"[WARN] 성공 trial이 {n_success}/{target_success}개만 확보되었습니다. (시도 {attempts}회)")
            break

    # 상위 top_k trial 반환
    best_trials = sorted(
        [t for t in study.best_trials], key=lambda t: t.value
    )[:top_k]
    return best_trials



# 1. 기존 코드
# if args.task != 'custom':
#     p_idx, labels_, cell_type, patient_id, data, cell_type_large = Covid_data(args)
# else:
#     p_idx, labels_, cell_type, patient_id, data, cell_type_large = Custom_data(args)

"""
# 2. covid, cardio를 위한 custom 추가 코드
# if args.task == 'custom_cardio':
#     p_idx, labels_, cell_type, patient_id, data, cell_type_large = Custom_data(args)
# elif args.task == 'custom_covid':
#     p_idx, labels_, cell_type, patient_id, data, cell_type_large = Custom_data(args)

# else:
#     p_idx, labels_, cell_type, patient_id, data, cell_type_large = Covid_data(args)


# 내부에서 랜덤하게 split을 생성 
# rkf = RepeatedKFold(n_splits=abs(args.n_splits), n_repeats=args.repeat * 100, random_state=args.seed)

# num = np.arange(len(p_idx))
# accuracy, aucs, cms, recalls, precisions = [], [], [], [], []
# iter_count = 0

# #  class 분포와 split 구성 확인
# from collections import Counter
# print(Counter(labels_))  # class 당, 전체 cell 단위 라벨 분포 # ex) Counter({1: 235252, 0: 185441, 2: 171996})
# patient_classes = [labels_[p[0]] for p in p_idx]
# print("환자 수 기준 클래스 분포:")
# print(Counter(patient_classes))
# # for i, p in enumerate(p_idx):
# #     print(f"Sample {i} - Class: {labels_[p[0]]}")


for train_index, test_index in rkf.split(num):
    print(f"🔍 Split #{iter_count + 1}")
    print(f"  → train_index 환자 수: {len(train_index)}")
    print(f"  → test_index 환자 수: {len(test_index)}")

    # 실제 환자 ID로 보기
    train_ids = [patient_id[p_idx[i][0]] for i in train_index]
    test_ids = [patient_id[p_idx[i][0]] for i in test_index]
    print(f"  → train 환자 ID: {train_ids}")
    print(f"  → test  환자 ID: {test_ids}")

    if args.n_splits < 0:
        temp_idx = train_index
        train_index = test_index
        test_index = temp_idx

    label_stat = [] #  train set에 포함된 환자들의 라벨 목록
    for idx in train_index:
        label_stat.append(labels_[p_idx[idx][0]])
    unique, cts = np.unique(label_stat, return_counts=True)
    # 훈련 데이터(train_index)에 클래스가 2개 이상 존재해야 학습을 진행한다.
    if len(unique) < 2 or (1 in cts): 
        # 클래스가 하나밖에 없음 → 불균형 → 스킵 
        # or 
        # 등장한 클래스 중 한 클래스의 환자 수가 1명밖에 안 됨 → 학습이 불안정해질 가능성이 매우 높기 때문에 skip
        continue
#     print(dict(zip(unique, cts)))
    
    # 원래 코드에는 test set의 클래스 불균형은 체크하지 않음
    ### ✅ test_index 클래스 확인 추가
    test_labels = [labels_[p_idx[i][0]] for i in test_index]
    if len(set(test_labels)) < 2:
        print(f"⚠️  Skipping split: test set has only one class -> {set(test_labels)}")
        continue

    kk = 0
    while True:
        train_index_, valid_index, ty, vy = train_test_split(train_index, label_stat, test_size=0.33,
                                                             random_state=args.seed + kk)
        if len(set(ty)) == 2 and len(set(vy)) == 2:
            break
        kk += 1

    train_index = train_index_
    len_valid = len(valid_index)
    _index = np.concatenate([valid_index, test_index])

    train_ids = []
    for i in train_index:
        # train_ids.append(patient_id[p_idx[i][0]])
        # pca False ; 수정 1
        train_ids.append(patient_id.iloc[p_idx[i][0]])

#     print(train_ids)

    x_train = []
    x_test = []
    x_valid = []
    y_train = []
    y_valid = []
    y_test = []
    id_train = []
    id_test = []
    id_valid = []
    data_augmented, train_p_idx, labels_augmented, cell_type_augmented = mixups(args, data,
                                                                                [p_idx[idx] for idx in train_index],
                                                                                labels_,
                                                                                cell_type)
    # mixup 실패 시 split 건너뛰기
    if data_augmented is None:
        print("⚠️ Skipping split due to insufficient mixup class diversity.")
        continue
    individual_train, individual_test = sampling(args, train_p_idx, [p_idx[idx] for idx in _index], labels_,
                                                 labels_augmented, cell_type_augmented)
    for t in individual_train:
        id, label = [id_l[0] for id_l in t], [id_l[1] for id_l in t]
        x_train += [ii for ii in id]
        y_train += (label)
        id_train += (id)

    temp_idx = np.arange(len(_index))
    for t_idx in temp_idx[len_valid:]:
        id, label = [id_l[0] for id_l in individual_test[t_idx]], [id_l[1] for id_l in individual_test[t_idx]]
        x_test.append([ii for ii in id])
        y_test.append(label[0])
        id_test.append(id)
    for t_idx in temp_idx[:len_valid]:
        id, label = [id_l[0] for id_l in individual_test[t_idx]], [id_l[1] for id_l in individual_test[t_idx]]
        x_valid.append([ii for ii in id])
        y_valid.append(label[0])
        id_valid.append(id)
    x_train, x_valid, x_test, y_train, y_valid, y_test = x_train, x_valid, x_test, np.array(y_train).reshape([-1, 1]), \
                                                         np.array(y_valid).reshape([-1, 1]), np.array(y_test).reshape(
        [-1, 1])
    auc, acc, cm, recall, precision = train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test,
                                            data_augmented, data)
    aucs.append(auc)
    accuracy.append(acc)
    cms.append(cm)
    recalls.append(recall)
    precisions.append(precision)
    iter_count += 1
    if iter_count == abs(args.n_splits) * args.repeat:
        break
    print(f"✅ Total valid splits used: {iter_count}")

    del data_augmented



# print("="*33)
# print("=== Final Evaluation (average across all splits) ===")
# print("="*33)

# print("Best performance: Test ACC %f,   Test AUC %f,   Test Recall %f,   Test Precision %f" % (np.average(accuracy), np.average(aucs), np.average(recalls), np.average(precisions)))

# print("=================================")
# print("=== 저희 논문용 Final Evaluation (average across all splits) ===")
# print("=================================")
# print(f"Best performance: "
#       f"Test ACC {np.mean(accuracy):.6f}+-{np.std(accuracy):.6f}, "
#       f"Test AUC {np.mean(aucs):.6f}+-{np.std(aucs):.6f}, "
#       f"Test Recall {np.mean(recalls):.6f}+-{np.std(recalls):.6f}, "
#       f"Test Precision {np.mean(precisions):.6f}+-{np.std(precisions):.6f}")

####################################
######## Only for repeat > 1 #######
####################################
# accuracy = np.array(accuracy).reshape([-1, args.repeat]).mean(0)
# aucs = np.array(aucs).reshape([-1, args.repeat]).mean(0)
# recalls = np.array(recalls).reshape([-1, args.repeat]).mean(0)
# precisions = np.array(precisions).reshape([-1, args.repeat]).mean(0)
# ci_1 = st.t.interval(alpha=0.95, df=len(accuracy) - 1, loc=np.mean(accuracy), scale=st.sem(accuracy))[1] - np.mean(accuracy)
# ci_2 = st.t.interval(alpha=0.95, df=len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs))[1] - np.mean(aucs)
# ci_3 = st.t.interval(alpha=0.95, df=len(recalls) - 1, loc=np.mean(recalls), scale=st.sem(recalls))[1] - np.mean(recalls)
# ci_4 = st.t.interval(alpha=0.95, df=len(precisions) - 1, loc=np.mean(precisions), scale=st.sem(precisions))[1] - np.mean(precisions)
# print("ci: ACC ci %f,   AUC ci %f,   Recall ci %f,   Precision ci %f" % (ci_1, ci_2, ci_3, ci_4))

# print(np.average(cms, 0))
# print(patient_summary)
# print(stats)
# print(stats_id)

"""
# 사전 생성된 split 파일을 기반으로 고정된 train/test 데이터셋으로 실험을 수행하기 위해 (2번코드) # => 주석처리

# 3. for loop 직접 구성 (repeat × fold)

per_repeat = []   # 각 repeat의 요약 기록용 리스트

for repeat in range(args.repeat):
    fold_aucs, accuracy, cms, recalls, precisions = [], [], [], [], []
    iter_count = 0
    for fold in range(args.n_splits):
        print(f"🔁 Repeat {repeat}, Fold {fold}")
        # train_path = f"/data/project/kim89/0805_data/repeat_{repeat}/fold_{fold}_train.h5ad"
        # test_path = f"/data/project/kim89/0805_data/repeat_{repeat}/fold_{fold}_test.h5ad"

        train_path = f"{args.dataset}/repeat_{repeat}/fold_{fold}_train.h5ad"
        test_path = f"{args.dataset}/repeat_{repeat}/fold_{fold}_test.h5ad"

        train_data = scanpy.read_h5ad(train_path)
        test_data = scanpy.read_h5ad(test_path)

        train_p_index, train_labels, train_cell_type, train_patient_id, train_origin = Custom_data_from_loaded(train_data, args)
        test_p_index, test_labels, test_cell_type, test_patient_id, test_origin = Custom_data_from_loaded(test_data, args)

        labels_ = np.concatenate([train_labels, test_labels])


        print(f"🔍 Split #{iter_count + 1}")
        print(f"  → train_p_index 환자 수 (환자 단위로 묶인 index list): {len(train_p_index)}")
        print(f"  → test_p_index 환자 수  (환자 단위로 묶인 index list): {len(test_p_index)}")
        # train_p_index_ 안에는 **"환자 단위 묶음"**이 들어있습니다.
        # 즉, 각 원소가 np.array([...]) 형태로, 한 환자에 속하는 모든 셀의 인덱스를 담고 있는 거예요.
        # 그래서 86629, 86630, ... 같은 숫자는 실제 "셀 번호(index)"일 뿐, 환자 ID가 아닙니다.

        # 실제 환자 ID로 보기
        train_ids = [train_patient_id[idx[0]] for idx in train_p_index]
        test_ids = [test_patient_id[idx[0]] for idx in test_p_index]
        print(f"  → train 환자 ID: {train_ids}")
        print(f"  → test  환자 ID: {test_ids}")

        # 각 환자의 ID와 label 함께 출력
        print("  → train 환자 ID 및 라벨:")
        for idxs in train_p_index:
            idx = idxs[0]
            print(f"    ID: {train_patient_id[idx]}, Label: {train_labels[idx]}")

        print("  → test 환자 ID 및 라벨:")
        for idxs in test_p_index:
            idx = idxs[0]
            print(f"    ID: {test_patient_id[idx]}, Label: {test_labels[idx]}")

        p_idx = train_p_index + test_p_index # 이어붙이기(concatenation) 로 동작
        print("전체 데이터 index 길이", len(p_idx))    

        # if args.n_splits < 0:
        #     temp_idx = train_p_index
        #     train_p_index = test_p_index
        #     test_p_index = temp_idx

        # train_labels: 전체 셀 단위 라벨 (186,636개)
        # 환자 단위 라벨만 뽑기
        label_stat = [labels_[idx[0]] for idx in train_p_index] #  train set에 포함된 환자들의 라벨 목록
        print("label_stat (train 환자 라벨 목록) 갯수", len(label_stat))
        # label_stat = []
        # for idx in train_p_index:
        #     label_stat.append(labels_[p_idx[idx][0]])

        unique, cts = np.unique(label_stat, return_counts=True)
    

        # 훈련 데이터(train_p_index)에 클래스가 2개 이상 존재해야 학습을 진행한다.
        if len(unique) < 2 or (1 in cts): 
            # 클래스가 하나밖에 없음 → 불균형 → 스킵 
            # or 
            # 등장한 클래스 중 한 클래스의 환자 수가 1명밖에 안 됨 → 학습이 불안정해질 가능성이 매우 높기 때문에 skip
            print("훈련 데이터(train_p_index)에 클래스가 2개 이상 존재해야 학습을 진행", flush=True)
            continue
    #     print(dict(zip(unique, cts)))
        
        # 원래 코드에는 test set의 클래스 불균형은 체크하지 않음
        # ### ✅ test_p_index 클래스 확인 추가
        # test_label_stat = [labels_[idx[0]] for idx in test_p_index]
        # if len(set(test_label_stat)) < 2:
        #     print(f"⚠️  Skipping split: test set has only one class -> {set(test_label_stat)}")
        #     continue

        # train_data에서 환자 단위로, train과 validation 나누기
        print("train_data에서 환자 단위로, train과 validation 나누기")
        print("기존 train_p_index",len(train_p_index))
        print("기존 (train) label_stat",len(label_stat))

        # 0) 최소 클래스 샘플 수 점검 (stratify가 요구)
        cts = Counter(label_stat)
        min_cls = min(cts.values())
        if min_cls < 2:
            print(f"⚠️ 최소 클래스 수가 2 미만: {cts}. 이 split은 건너뜁니다.")
            continue

        # 1) stratify로 한 번에 해결 (while 불필요)
        try:
            train_p_index_, valid_p_index, ty, vy = train_test_split(
                train_p_index, 
                # label_stat,
                # [train_labels[idx[0]] for idx in train_p_index],  # 해당 환자 라벨 목록
                label_stat,        # 환자 단위 라벨
                test_size=0.33,
                random_state=args.seed
                # stratify=label_stat
                # stratify= [train_labels[idx[0]] for idx in train_p_index]  # 해당 환자 라벨 목록 
                )
        except ValueError as e:
            # 2) 실패 시 test_size 축소해서 한 번 더 시도
            print(f"⚠️ stratify 실패({e}). test_size=0.2로 재시도합니다.")
            train_p_index_, valid_p_index, ty, vy = train_test_split(
                train_p_index, train_labels,
                test_size=0.2,
                random_state=args.seed,
                # stratify=label_stat
            )

        # 3) (안전망) 분할 후 클래스 전부 포함됐는지 확인
        classes = set(label_stat)
        print("train-vali 분할 후! ")
        print("train y 의 classes :",set(ty), "valid y 의 classes :",set(vy), "label_stat 의 classes :",classes)
        if not (set(ty) == classes and set(vy) == classes):
            print(f"⚠️ 분할 후 일부 클래스가 빠짐. (train={set(ty)}, val={set(vy)}) 이 split은 건너뜁니다.")
            continue

        print("train_p_index_",len(train_p_index_), flush=True)
        print("valid_p_index",len(valid_p_index), flush=True)
        print("test_p_index",len(test_p_index), flush=True)

        train_p_index = train_p_index_
        print("→ train 환자 ID 및 라벨:")
        print("   총 개수:", len(train_p_index))
        for idxs in train_p_index:
            idx0 = idxs[0]
            print(f"   환자ID={train_patient_id[idx0]}, Label={train_labels[idx0]}, 셀개수={len(idxs)}")

        print("→ valid 환자 ID 및 라벨:")
        print("   총 개수:", len(valid_p_index))
        for idxs in valid_p_index:
            idx0 = idxs[0]
            print(f"   환자ID={train_patient_id[idx0]}, Label={train_labels[idx0]}, 셀개수={len(idxs)}")


        # ✅ test는 test_patient_id/test_labels로!
        print("→ test 환자 ID 및 라벨:")
        print("   총 개수:", len(test_p_index))
        for idxs in test_p_index:
            idx0 = idxs[0]
            print(f"   환자ID={test_patient_id[idx0]}, Label={test_labels[idx0]}, 셀개수={len(idxs)}")


        train_p_index = train_p_index_
        len_valid = len(valid_p_index)
        # _index = np.concatenate([valid_p_index, test_p_index])


        # train_ids = []
        # for i in train_p_index:
        #     train_ids.append(patient_id.iloc[p_idx[i][0]])

    #     print(train_ids)

        x_train = []
        x_test = []
        x_valid = []
        y_train = []
        y_valid = []
        y_test = []
        id_train = []
        id_test = []
        id_valid = []

        train_cell_type = pd.Series([
            ct if isinstance(ct, str) else "Unknown"
            for ct in train_cell_type
        ])

        print("✅ Checking cell_type before mixups...")
        print("Unique types:", set([type(x) for x in train_cell_type]))
        print("Example values (first 10):", list(train_cell_type[:10]))
        print("NaN exists?", any([isinstance(x, float) and np.isnan(x) for x in train_cell_type]))
        # 1. Series에 NaN이 있는지 확실히 확인
        print("🔍 isna count:", pd.Series(train_cell_type).isna().sum())

        # 2. set 안에 float (NaN)가 섞여 있는지 확인
        print("🧪 Types in set(cell_type):", set([type(x) for x in set(train_cell_type)]))

        data = np.concatenate([train_origin, test_origin])
        patient_id_all = np.concatenate([np.array(train_patient_id), np.array(test_patient_id)])

        print("전체 데이터 길이", len(data))
        
        def to_series_1d(x):
            # numpy / list / Series 모두 1차원 Series로 통일
            if isinstance(x, pd.Series):
                return x.reset_index(drop=True)
            return pd.Series(np.ravel(x))

        train_ct = to_series_1d(train_cell_type)
        test_ct  = to_series_1d(test_cell_type)

        # 행 방향 이어붙이기 + 인덱스 초기화
        cell_type = pd.concat([train_ct, test_ct], ignore_index=True)
        cell_type = cell_type.astype("string").fillna("Unknown")
        print("전체 cell type 길이", cell_type)
        print("cell type의 NaN 개수 체크",cell_type.isna().sum())
        data_augmented, train_p_idx, labels_augmented, cell_type_augmented = mixups(args, data,
                                                                            train_p_index,
                                                                            labels_,
                                                                            cell_type)
        # main.py - repeat×fold 루프 내부에서, train/valid/test 분할을 만든 뒤

        # 4-1) 튜닝용(=valid만 평가 후보) 데이터 패킹
        # mixups/sampling 전에 이미 만든 train_p_index, valid_p_index, test_p_index, data/labels_가 있다고 가정
        # (현재 코드에서 바로 위까지 준비되는 구조입니다. 예: eval_p_index 구성부 등) 

        # ---> (A) 먼저 "valid만" 평가 후보로 해서 sampling
        eval_only_valid = valid_p_index  # test 미포함
        individual_train_A, individual_eval_A = sampling(
            args,
            train_p_idx,         # mixups() 결과
            eval_only_valid,     # valid만
            labels_, labels_augmented, cell_type_augmented
        )

        # numpy로 변환 (튜닝 objective에서 train 호출할 수 있게)
        x_train_A, y_train_A, id_train_A = [], [], []
        x_valid_A, y_valid_A, id_valid_A = [], [], []
        for t in individual_train_A:
            id_, label_ = [id_l[0] for id_l in t], [id_l[1] for id_l in t]
            x_train_A += [ii for ii in id_]
            y_train_A += label_
            id_train_A += id_
        for t in individual_eval_A:
            id_, label_ = [id_l[0] for id_l in t], [id_l[1] for id_l in t]
            x_valid_A.append([ii for ii in id_])
            y_valid_A.append(label_[0])
            id_valid_A.append(id_)

        prep_A = dict(
            x_train=x_train_A, x_valid=x_valid_A,
            y_train=np.array(y_train_A).reshape([-1,1]),
            y_valid=np.array(y_valid_A).reshape([-1,1]),
            id_train=id_train_A, id_valid=id_valid_A,
            data_augmented=data_augmented, data=data
        )

        # (B) 최종 평가용: valid + test 후보를 한 번에 넘겨서 sampling
        offset = train_origin.shape[0]
        test_p_index_global = [idx + offset for idx in test_p_index]
        eval_valid_plus_test = valid_p_index + test_p_index_global

        individual_train_B, individual_eval_B = sampling(
            args,
            train_p_idx,                   # mixups() 결과
            eval_valid_plus_test,          # valid + test 후보
            labels_, labels_augmented, cell_type_augmented
        )

        # numpy/list로 변환
        x_train_B, y_train_B, id_train_B = [], [], []
        for t in individual_train_B:
            ids  = [id_l[0] for id_l in t]
            lbls = [id_l[1] for id_l in t]
            x_train_B += [ii for ii in ids]
            y_train_B += lbls
            id_train_B += ids

        # ★ 핵심: 앞부분은 valid, 뒷부분은 test로 "다시 분리"
        len_valid = len(valid_p_index)
        x_valid_B, y_valid_B, id_valid_B = [], [], []
        x_test_B,  y_test_B,  id_test_B  = [], [], []

        for t_idx, t in enumerate(individual_eval_B):
            ids  = [id_l[0] for id_l in t]
            lbls = [id_l[1] for id_l in t]
            if t_idx < len_valid:
                x_valid_B.append([ii for ii in ids])
                y_valid_B.append(lbls[0])
                id_valid_B.append(ids)
            else:
                x_test_B.append([ii for ii in ids])
                y_test_B.append(lbls[0])
                id_test_B.append(ids)

        # 배열 모양 맞추기
        y_train_B = np.array(y_train_B).reshape([-1, 1])
        y_valid_B = np.array(y_valid_B).reshape([-1, 1])
        y_test_B  = np.array(y_test_B ).reshape([-1, 1])


        if args.heads == 4 and args.emb_dim >= 64:
            args.batch_size = min(args.batch_size, 8)


        # (1) Optuna로 튜닝 ; train과 validation만 들어감
        best_trials = optuna_tune_one_fold(prep_A, n_trials=3, top_k=args.top_k) ### n_trials=3 ; 3가지의 hyperparameter tuning만 수행함

        # (2) 선택된 trial(1~3개) 각각으로 최종 평가(test 포함)
        # data_for_training = data_augmented if use_aug else data
        # (안전) top-k 후보가 비어 있을 수 있으므로 먼저 체크
        if len(best_trials) == 0:
            print("[WARN] 튜닝 결과 유효한 trial이 없어 이 fold를 건너뜁니다.")
            fold_aucs.append(np.nan)
            accuracy.append(np.nan)
            recalls.append(np.nan)
            precisions.append(np.nan)
            cms.append(None)
        else:
            # ⬇️ 여기서 best_fold를 '없음'으로 초기화 (또는 auc=-inf로 초기화)
            best_fold = None  # or: {"auc": -float("inf")}

            for t in best_trials:
                hp = t.params
                print("선택된 trial params:", hp)
                # (1) hp → args 주입
                args.learning_rate = hp["learning_rate"]
                args.weight_decay  = hp["weight_decay"]
                args.epochs        = hp["epochs"]
                args.heads         = hp["heads"]
                args.dropout       = hp["dropout"]
                args.emb_dim       = hp["emb_dim"]
                args.augment_num   = hp["augment_num"]
                args.pca           = hp["pca"]

                # (2) OOM 사전 가드
                orig_bs = args.batch_size
                if args.heads == 4 and args.emb_dim >= 64:
                    args.batch_size = min(args.batch_size, 8)

                try:
                    data_for_training = data_augmented  # 증강 ON이면 mixup 결과, OFF 설계면 data로 교체
                    def _call():
                        return train(
                            x_train_B, x_valid_B, x_test_B,
                            y_train_B, y_valid_B, y_test_B,
                            id_train_B, id_test_B,
                            data_for_training, data, eval_test=True
                        )

                    # (3) OOM 래퍼로 감싸 실행
                    auc, acc, cm, recall, precision, _ = train_with_oom_retry(_call, backoff=(8, 4, 2, 1))

                    # (4) test AUC가 더 크면 best 후보로 교체
                    if (best_fold is None) or (auc > best_fold["auc"]):
                        best_fold = {
                            "auc": auc,
                            "acc": acc,
                            "cm": cm,
                            "recall": recall,
                            "precision": precision,
                        }

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[WARN] OOM on final test for params {hp} → skip this candidate.")
                        continue
                    else:
                        raise
                finally:
                    args.batch_size = orig_bs  # 원복

            # (5) 모든 후보가 OOM 등으로 실패했을 가능성 처리
            if best_fold is None:
                fold_aucs.append(np.nan)
                accuracy.append(np.nan)
                recalls.append(np.nan)
                precisions.append(np.nan)
                cms.append(None)
            else:
                # ★ 여기서만 집계 — ‘가장 좋은 후보’의 test 지표
                fold_aucs.append(best_fold["auc"])
                accuracy.append(best_fold["acc"])
                cms.append(best_fold["cm"])
                recalls.append(best_fold["recall"])
                precisions.append(best_fold["precision"])


        iter_count += 1
        if iter_count == abs(args.n_splits) * args.repeat:
            break
        print(f"✅ Total valid splits used: {iter_count}")

        # === FOLD CLEANUP: 다음 fold로 넘어가기 전에 큰 객체 정리 ===
        # 1) 큰 파이썬 객체/배열은 직접 None 할당로 끊어주세요.
        #    (아래 변수명은 여러분 코드에 맞게 조정)
        x_train_A = x_valid_A = y_train_A = y_valid_A = id_train_A = id_valid_A = None
        x_train_B = x_valid_B = x_test_B = None
        y_train_B = y_valid_B = y_test_B = None
        id_train_B = id_test_B = None
        individual_train_A = individual_eval_A = None
        individual_train_B = individual_eval_B = None

        # AnnData / 원본 배열
        train_data = test_data = None
        data = data_augmented = None
        train_p_idx = labels_augmented = cell_type_augmented = None

        # (Optuna 객체도 fold별로 생성했다면)
        try:
            study = None
        except NameError:
            pass

        # 2) 가비지 컬렉션 + CUDA 캐시 비우기
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

        
    # 🔽 Repeat 단위 AUC/지표 출력
    rep_auc_mean = float(np.nanmean(fold_aucs)) if len(fold_aucs) else float("nan")
    rep_auc_std  = float(np.nanstd(fold_aucs))  if len(fold_aucs) else float("nan")
    rep_acc_mean = float(np.nanmean(accuracy))  if len(accuracy) else float("nan")
    rep_rec_mean = float(np.nanmean(recalls))   if len(recalls) else float("nan")
    rep_pre_mean = float(np.nanmean(precisions)) if len(precisions) else float("nan")
    nan_count    = int(np.count_nonzero(np.isnan(fold_aucs)))
    n_folds      = int(len(fold_aucs))

    print(f"\n📌 Repeat {repeat}: 평균 AUC = {rep_auc_mean:.4f}, 표준편차 = {rep_auc_std:.4f}")
    print(f"Test ACC 평균 {rep_acc_mean:.6f}, Test Recall 평균 {rep_rec_mean:.6f}, Test Precision 평균 {rep_pre_mean:.6f}")
    print("fold_aucs =", fold_aucs)
    print(f"NaN 개수: {nan_count} / 전체 {n_folds}개\n")

    # ⬇️ 이 repeat의 요약을 저장 (나중에 전체 요약에서 사용)
    per_repeat.append({
        "repeat": repeat,
        "auc_mean": rep_auc_mean,
        "auc_std": rep_auc_std,
        "acc_mean": rep_acc_mean,
        "recall_mean": rep_rec_mean,
        "precision_mean": rep_pre_mean,
        "nan_count": nan_count,
        "n_folds": n_folds,
    })

    # === REPEAT CLEANUP === (옵션)
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

# ⬇️ 모든 Repeat가 끝난 뒤, 각 Repeat의 값과 최종 평균 출력
print("\n================ 전체 Repeat 요약 ================")
for s in per_repeat:
    print(
        f"Repeat {s['repeat']}: "
        f"AUC {s['auc_mean']:.4f} ± {s['auc_std']:.4f} | "
        f"ACC {s['acc_mean']:.4f} | "
        f"Recall {s['recall_mean']:.4f} | "
        f"Precision {s['precision_mean']:.4f} | "
        f"NaN {s['nan_count']}/{s['n_folds']}"
    )

# 최종 결과: “5개의 fold를 평균한 각 Repeat의 AUC 평균들”의 평균(=macro 평균)
if len(per_repeat):
    final_auc_mean = float(np.nanmean([s["auc_mean"] for s in per_repeat]))
    final_auc_std  = float(np.nanstd([s["auc_mean"]  for s in per_repeat]))
    print(f"\n🏁 최종 결과 (각 Repeat의 AUC 평균의 평균): {final_auc_mean:.4f} ± {final_auc_std:.4f}")
else:
    print("\n🏁 최종 결과를 계산할 Repeat 요약이 없습니다.")

