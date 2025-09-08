from sklearn import metrics
from sklearn.metrics import accuracy_score
import scipy.stats as st
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
parser.add_argument('--augment_num', type=int, default=0) # Mixup된 새로운 가짜 샘플을 몇 개 생성할지
parser.add_argument('--alpha', type=float, default=1.0) # mixup의 비율 (Beta 분포 파라미터)
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--all', type=int, default=1)
# all == 0:
    # sample_cells 만큼 랜덤 샘플링 (np.random.choice)
# all == 1
    # 샘플링을 건너뛰고 '해당 환자(혹은 라벨)의 모든 셀'을 그대로 사용
parser.add_argument('--min_size', type=int, default=6000)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--pca', type=_str2bool, default=True)
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

def safe_gather(data_mat, index_tensor):
    # index_tensor: LongTensor (B, N) 또는 (N,)
    idx = index_tensor.clone()
    idx[idx < 0] = 0  # -1 패딩은 0으로 치환해 안전 인덱싱
    return torch.from_numpy(data_mat[idx.cpu().numpy()])


def train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, data_augmented, data):
    dataset_1 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='train')
    dataset_2 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='test')
    dataset_3 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='val')
    train_loader = torch.utils.data.DataLoader(dataset_1, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=dataset_1.collate)
    test_loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False, collate_fn=dataset_2.collate)
    valid_loader = torch.utils.data.DataLoader(dataset_3, batch_size=1, shuffle=False, collate_fn=dataset_3.collate)

    print("👉 train_loader 길이 (샘플수/batch크기):", len(train_loader))
    print("👉 test_loader 길이:", len(test_loader))
    print("👉 valid_loader 길이:", len(valid_loader))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = data_augmented[0].shape[-1]

    output_class = len(set(labels_))
    if (output_class == 2):
        output_class = 1

    if args.task == 'custom_covid':
        output_class = 1
    if args.task == 'custom_cardio':
        output_class = 3
    if args.task == 'custom_kidney':
        output_class = 3

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
            id_ = batch[2][0]

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

            else:
                # ===== Multiclass =====
                # logits을 [1, C]로 축약
                if logits.dim() == 3:            # [B=1, N, C]
                    logits_bag = logits.mean(dim=1)      # [1, C]
                elif logits.dim() == 2:          # [N, C] 또는 [1, C]
                    if logits.shape[0] > 1:      # [N, C]이면 N 평균
                        logits_bag = logits.mean(dim=0, keepdim=True)   # [1, C]
                    else:
                        logits_bag = logits                         # [1, C]
                else:
                    raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

                # 정답 (스칼라)
                # y_ = batch[1]  # torch.FloatTensor shape [1,1] (DataLoader collate에서 보장)
                # y_true = int(y_.detach().cpu().reshape(-1)[0].item())  # ✅ 안전
                
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
            recall = precision = np.nan

        # 로그
        for i in range(len(pred)):
            print(f"{test_ids[i]} -- true: {label_dict[true[i]]} -- pred: {label_dict[pred[i]]}")

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

    return test_auc, test_acc, cm, recall, precision


if args.model != 'Transformer':
    args.repeat = 60

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

        # 1) Train 쪽 mixup/증강
        # if args.augment_num > 0:
        #     print("data augment 실행 함")
        #     data_augmented, train_p_index_aug, labels_aug, cell_type_aug = mixups(
        #         args, train_origin, train_p_index_, train_labels, train_cell_type
        #     )
        #     if data_augmented is None:
        #         print("⚠️ Skipping due to insufficient classes for mixup")
        #         continue
        # else:
        #     print("data augment 실행 안 함")
        #     data_augmented = train_origin
        #     train_p_index_aug = train_p_index_
        #     labels_aug = train_labels
        #     cell_type_aug = train_cell_type

        data = np.concatenate([train_origin, test_origin])
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

        _index = valid_p_index + test_p_index  # ✅ 리스트끼리 결합
                                                        
        # 1) train/test 결합 길이 기준으로 오프셋 계산
        offset = train_origin.shape[0]

        # 2) test 쪽 인덱스들에 오프셋 적용
        test_p_index_global = [idx + offset for idx in test_p_index]

        # 3) valid + test를 그대로 합쳐서 평가 후보 만들기
        eval_p_index = valid_p_index + test_p_index_global

        # 4) mixup 결과를 사용해 train 쪽은 train_p_index_aug를 쓰고,
        #    test 쪽은 eval_p_index를 그대로 sampling()에 전달
        individual_train, individual_test = sampling(
            args,
            train_p_idx,      # mixups()가 돌려준 train 쪽(증강 포함) 환자별 셀 인덱스 배열 리스트
            eval_p_index,           # 이미 "배열 리스트" 형태 → p_idx[...]로 다시 인덱싱 금지
            labels_,                # train+test 라벨을 concat 한 벡터 (배열 인덱스가 전역 기준이어야 함)
            labels_augmented,             # mixup 후 라벨
            cell_type_augmented           # mixup 후 셀타입
        )
  
        # # 평가용 인덱스는 valid + test로 합쳐서 sampling (scRAT 구조상 하나로 묶어서 sampling)
        # eval_p_index = valid_p_index + test_p_index
        # print("eval_p_index len: ",len(eval_p_index))

        # individual_train, individual_eval = sampling(
        #     args,
        #     train_p_index_aug,
        #     eval_p_index,
        #     train_labels,
        #     labels_aug,
        #     cell_type_aug
        # )
        # print("individual_train", len(individual_train))
        # print("individual_eval", len(individual_eval))

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
            

        # n_valid = len(valid_p_index)
        # for i in range(len(eval_p_index)):
        #     ids, labels = [x[0] for x in individual_eval[i]], [x[1] for x in individual_eval[i]]
        #     if i < n_valid:
        #         x_valid.append(ids)
        #         y_valid.append(labels[0])
        #         id_valid.append(ids)
        #     else:
        #         x_test.append(ids)
        #         y_test.append(labels[0])
        #         id_test.append(ids)


        x_train, x_valid, x_test, y_train, y_valid, y_test = x_train, x_valid, x_test, np.array(y_train).reshape([-1, 1]), \
                                                            np.array(y_valid).reshape([-1, 1]), np.array(y_test).reshape([-1, 1])
        print("train data의 x, y, id 길이:", len(x_train), len(y_train), len(id_train))
        print("valid data의 x, y, id 길이:", len(x_valid), len(y_valid), len(id_valid))
        print("test data의 x, y, id 길이:", len(x_test), len(y_test), len(id_test))

        # 4) 학습/검증/테스트 실행
        # auc, acc, cm, recall, precision = train(
        #     x_train, x_valid, x_test,
        #     y_train, y_valid, y_test,
        #     id_train, id_test,
        #     data_augmented=data_augmented,  # numpy array
        #     data=train_origin             # or full data if needed
        # )
        auc, acc, cm, recall, precision = train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test,
                                                data_augmented, data)
        

        fold_aucs.append(auc)
        accuracy.append(acc)
        cms.append(cm)
        recalls.append(recall)
        precisions.append(precision)
        iter_count += 1
        if iter_count == abs(args.n_splits) * args.repeat:
            break
        print(f"✅ Total valid splits used: {iter_count}")

        del data_augmented
        
    # 🔽 Repeat 단위 AUC 출력 추가
    print(f"\n📌 Repeat {repeat}: 평균 AUC = {np.nanmean(fold_aucs):.4f}, 표준편차 = {np.nanstd(fold_aucs):.4f}")
    print(
      f"Test ACC 평균 {np.nanmean(accuracy):.6f}, "
      f"Test Recall 평균 {np.nanmean(recalls):.6f}, "
      f"Test Precision 평균 {np.nanmean(precisions):.6f}")
    
    print("fold_aucs =", fold_aucs)
    nan_count = np.count_nonzero(np.isnan(fold_aucs))
    print(f"NaN 개수: {nan_count} / 전체 {len(fold_aucs)}개\n\n")


