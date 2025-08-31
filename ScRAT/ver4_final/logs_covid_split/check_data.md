# 1. 원본(기본) 데이터
라벨별 그룹 개수 {1: 28, 0: 12}
라벨별 그룹 개수 {1: 7, 0: 3}

  → train_p_index 환자 수 (환자 단위로 묶인 index list): 40
  → test_p_index 환자 수  (환자 단위로 묶인 index list): 10

    → train 환자 ID 및 라벨:
    ID: COVID19_Participant10, Label: 1
    ID: COVID19_Participant12, Label: 1
    ID: COVID19_Participant13, Label: 1
    ID: COVID19_Participant14, Label: 1
    ID: COVID19_Participant15, Label: 1
    ID: COVID19_Participant17, Label: 1
    ID: COVID19_Participant19, Label: 1
    ID: COVID19_Participant2, Label: 1
    ID: COVID19_Participant20, Label: 1
    ID: COVID19_Participant21, Label: 1
    ID: COVID19_Participant22, Label: 1
    ID: COVID19_Participant23, Label: 1
    ID: COVID19_Participant26, Label: 1
    ID: COVID19_Participant27, Label: 1
    ID: COVID19_Participant29, Label: 1
    ID: COVID19_Participant3, Label: 1
    ID: COVID19_Participant31, Label: 1
    ID: COVID19_Participant32, Label: 1
    ID: COVID19_Participant35, Label: 1
    ID: COVID19_Participant36, Label: 1
    ID: COVID19_Participant37, Label: 1
    ID: COVID19_Participant38, Label: 1
    ID: COVID19_Participant4, Label: 1
    ID: COVID19_Participant40, Label: 1
    ID: COVID19_Participant5, Label: 1
    ID: COVID19_Participant6, Label: 1
    ID: COVID19_Participant7, Label: 1
    ID: COVID19_Participant9, Label: 1
    ID: Control_Participant1, Label: 0
    ID: Control_Participant10, Label: 0
    ID: Control_Participant11, Label: 0
    ID: Control_Participant12, Label: 0
    ID: Control_Participant13, Label: 0
    ID: Control_Participant15, Label: 0
    ID: Control_Participant2, Label: 0
    ID: Control_Participant3, Label: 0
    ID: Control_Participant4, Label: 0
    ID: Control_Participant5, Label: 0
    ID: Control_Participant6, Label: 0
    ID: Control_Participant8, Label: 0
  → test 환자 ID 및 라벨:
    ID: COVID19_Participant11, Label: 1
    ID: COVID19_Participant18, Label: 1
    ID: COVID19_Participant25, Label: 1
    ID: COVID19_Participant28, Label: 1
    ID: COVID19_Participant30, Label: 1
    ID: COVID19_Participant39, Label: 1
    ID: COVID19_Participant8, Label: 1
    ID: Control_Participant14, Label: 0
    ID: Control_Participant7, Label: 0
    ID: Control_Participant9, Label: 0




# 2. train_data에서 환자 단위로, train과 validation 
train_p_index_ 26
valid_p_index 14
test_p_index 10

→ train 환자 ID 및 라벨:
   총 개수: 26
   환자ID=Control_Participant10, Label=0, 셀개수=1645
   환자ID=Control_Participant15, Label=0, 셀개수=528
   환자ID=COVID19_Participant19, Label=1, 셀개수=816
   환자ID=COVID19_Participant38, Label=1, 셀개수=51
   환자ID=COVID19_Participant10, Label=1, 셀개수=18
   환자ID=Control_Participant4, Label=0, 셀개수=440
   환자ID=COVID19_Participant14, Label=1, 셀개수=382
   환자ID=COVID19_Participant2, Label=1, 셀개수=329
   환자ID=COVID19_Participant40, Label=1, 셀개수=381
   환자ID=COVID19_Participant13, Label=1, 셀개수=730
   환자ID=COVID19_Participant20, Label=1, 셀개수=2033
   환자ID=Control_Participant2, Label=0, 셀개수=41
   환자ID=COVID19_Participant27, Label=1, 셀개수=93
   환자ID=Control_Participant6, Label=0, 셀개수=131
   환자ID=Control_Participant5, Label=0, 셀개수=717
   환자ID=COVID19_Participant15, Label=1, 셀개수=1003
   환자ID=Control_Participant8, Label=0, 셀개수=393
   환자ID=Control_Participant11, Label=0, 셀개수=1164
   환자ID=COVID19_Participant22, Label=1, 셀개수=865
   환자ID=COVID19_Participant31, Label=1, 셀개수=410
   환자ID=COVID19_Participant26, Label=1, 셀개수=814
   환자ID=COVID19_Participant17, Label=1, 셀개수=843
   환자ID=COVID19_Participant6, Label=1, 셀개수=84
   환자ID=COVID19_Participant4, Label=1, 셀개수=1680
   환자ID=COVID19_Participant37, Label=1, 셀개수=668
   환자ID=COVID19_Participant3, Label=1, 셀개수=104
→ valid 환자 ID 및 라벨:
   총 개수: 14
   환자ID=Control_Participant13, Label=0, 셀개수=88
   환자ID=COVID19_Participant29, Label=1, 셀개수=573
   환자ID=COVID19_Participant35, Label=1, 셀개수=83
   환자ID=COVID19_Participant32, Label=1, 셀개수=1665
   환자ID=COVID19_Participant5, Label=1, 셀개수=89
   환자ID=COVID19_Participant21, Label=1, 셀개수=39
   환자ID=Control_Participant3, Label=0, 셀개수=1620
   환자ID=Control_Participant1, Label=0, 셀개수=414
   환자ID=COVID19_Participant9, Label=1, 셀개수=86
   환자ID=COVID19_Participant7, Label=1, 셀개수=409
   환자ID=Control_Participant12, Label=0, 셀개수=99
   환자ID=COVID19_Participant36, Label=1, 셀개수=200
   환자ID=COVID19_Participant12, Label=1, 셀개수=102
   환자ID=COVID19_Participant23, Label=1, 셀개수=307
→ test 환자 ID 및 라벨:
   총 개수: 10
   환자ID=COVID19_Participant11, Label=1, 셀개수=382
   환자ID=COVID19_Participant18, Label=1, 셀개수=227
   환자ID=COVID19_Participant25, Label=1, 셀개수=496
   환자ID=COVID19_Participant28, Label=1, 셀개수=1079
   환자ID=COVID19_Participant30, Label=1, 셀개수=116
   환자ID=COVID19_Participant39, Label=1, 셀개수=454
   환자ID=COVID19_Participant8, Label=1, 셀개수=462
   환자ID=Control_Participant14, Label=0, 셀개수=274
   환자ID=Control_Participant7, Label=0, 셀개수=762
   환자ID=Control_Participant9, Label=0, 셀개수=558


# 3. ======= sample mixup ... ============

# 4.samling() : bag()만듦

# 5. hyper parameter tuning 5번 시행
👉 train samples(학습 샘플(bag))=126, batch_size=16 -> steps(샘플수/batch크기)=8
👉 valid samples=14, batch_size=1 -> steps=14
👉 test  samples=0, batch_size=1 -> steps=0

# 6. test 결과
선택된 trial params: {'learning_rate': 0.001, 'epochs': 100, 'heads': 1, 'dropout': 0.5, 'weight_decay': 0.01, 'emb_dim': 8, 'augment_num': 100, 'pca': False}
👉 train samples(학습 샘플(bag))=126, batch_size=16 -> steps(샘플수/batch크기)=8
👉 valid samples=14, batch_size=1 -> steps=14
👉 test  samples=10, batch_size=1 -> steps=10

환자ID=COVID19_Participant11 -- true: 1 -- pred: 1
환자ID=COVID19_Participant18 -- true: 1 -- pred: 1
환자ID=COVID19_Participant25 -- true: 1 -- pred: 1
환자ID=COVID19_Participant28 -- true: 1 -- pred: 1
환자ID=COVID19_Participant30 -- true: 1 -- pred: 1
환자ID=COVID19_Participant39 -- true: 1 -- pred: 1
환자ID=COVID19_Participant8 -- true: 1 -- pred: 1
환자ID=Control_Participant14 -- true: 0 -- pred: 0
환자ID=Control_Participant7 -- true: 0 -- pred: 1
환자ID=Control_Participant9 -- true: 0 -- pred: 0

Best performance: Epoch 30, Loss 0.504947, Test ACC 0.900000, Test AUC 1.000000, Test Recall 1.000000, Test Precision 0.875000
Confusion Matrix:
 [2 1 0 7]

# +. 선택된 hyperparameter로, 기본 ScRAT 코드 (전혀 수정 안 한 원본)에 넣어서 돌려봤다.
