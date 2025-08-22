import numpy as np
import pickle
import scanpy
import pandas as pd


def Covid_data(args):
    if args.task == 'haniffa':
        id_dict = {'Critical ': 1, 'Death': -1, 'Severe': 1, 'nan': -1, 'LPS': 0, 'Non-covid': 0, 'Asymptomatic': 1,
                   'Mild': 1, 'Healthy': 0, 'Moderate': 1}

        if args.pca == True:
            with open('./data/Haniffa/Haniffa_X_pca.npy', 'rb') as f:
                origin = np.load(f)
        else:
            with open('./data/Haniffa/origin.npy', 'rb') as f:
                origin = np.load(f)

        a_file = open('./data/Haniffa/patient_id.pkl', "rb")
        patient_id = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/Haniffa/labels.pkl', "rb")
        labels = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/Haniffa/cell_type.pkl', "rb")
        cell_type = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/Haniffa/cell_type_large.pkl', "rb")
        cell_type_large = pickle.load(a_file)
        a_file.close()
    elif args.task == 'combat':
        id_dict = {'COVID_HCW_MILD': 1, 'COVID_CRIT': 1, 'COVID_MILD': 1, 'COVID_SEV': 1, 'COVID_LDN': 1, 'HV': 0,
                   'Flu': 0, 'Sepsis': 0}

        if args.pca == True:
            with open('./data/COMBAT/COMBAT_X_pca.npy', 'rb') as f:
                origin = np.load(f)
        else:
            with open('./data/COMBAT/origin.npy', 'rb') as f:
                origin = np.load(f)

        a_file = open('./data/COMBAT/patient_id.pkl', "rb")
        patient_id = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/COMBAT/labels.pkl', "rb")
        labels = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/COMBAT/cell_type.pkl', "rb")
        cell_type = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/COMBAT/cell_type_large.pkl', "rb")
        cell_type_large = pickle.load(a_file)
        a_file.close()
    else:
        id_dict = {}
        if args.task == 'severity':
            id_dict = {'mild/moderate': 0, 'severe/critical': 1, 'control': -1}
        elif args.task == 'stage':
            id_dict = {'convalescence': 0, 'progression': 1, 'control': -1}

        if args.pca == True:
            with open('./data/SC4/covid_pca.npy', 'rb') as f:
                origin = np.load(f)
        else:
            with open('./data/SC4/origin.npy', 'rb') as f:
                origin = np.load(f)

        a_file = open('./data/SC4/patient_id.pkl', "rb")
        patient_id = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/SC4/' + args.task + '_label.pkl', "rb")
        labels = pickle.load(a_file)
        a_file.close()

        if args.task == 'severity':
            a_file = open('./data/SC4/stage_label.pkl', "rb")
            stage_labels = pickle.load(a_file)
            a_file.close()

        a_file = open('./data/SC4/cell_type.pkl', "rb")
        cell_type = pickle.load(a_file)
        a_file.close()

        a_file = open('./data/SC4/cell_type_large.pkl', "rb")
        cell_type_large = pickle.load(a_file)
        a_file.close()

    labels_ = np.array(labels.map(id_dict))

    if args.task == 'severity':
        id_dict_ = {'convalescence': 0, 'progression': 1, 'control': 0}
        labels_stage = np.array(stage_labels.map(id_dict_))

    l_dict = {}
    indices = np.arange(origin.shape[0])
    p_ids = sorted(set(patient_id))
    p_idx = []

    if args.task == 'combat':
        top_class = []
        for tc in (
        'PB', 'CD4.TEFF.prolif', 'PLT', 'B.INT', 'CD8.TEFF.prolif', 'B.MEM', 'NK.cyc', 'RET', 'B.NAIVE', 'NK.mitohi'):
            top_class.append(indices[cell_type_large == tc])
        selected = np.concatenate(top_class)
    elif args.task == 'haniffa':
        top_class = []
        for tc in (
        'B_immature', 'C1_CD16_mono', 'CD4.Prolif', 'HSC_erythroid', 'RBC', 'Plasma_cell_IgG', 'pDC', 'Plasma_cell_IgA',
        'Platelets', 'Plasmablast'):
            top_class.append(indices[cell_type_large == tc])
        selected = np.concatenate(top_class)
    elif args.task == 'severity':
        top_class = []
        for tc in (
        'Macro_c3-EREG', 'Epi-Squamous', 'Neu_c5-GSTP1(high)OASL(low)', 'Epi-Ciliated', 'Neu_c3-CST7', 'Neu_c4-RSAD2',
        'Epi-Secretory', 'Mega', 'Neu_c1-IL1B', 'Macro_c6-VCAN', 'DC_c3-LAMP3', 'Neu_c6-FGF23', 'Macro_c2-CCL3L1',
        'Mono_c1-CD14-CCL3', 'Neu_c2-CXCR4(low)', 'B_c05-MZB1-XBP1', 'DC_c1-CLEC9A', 'Mono_c4-CD14-CD16'):
            top_class.append(indices[cell_type_large == tc])
        selected = np.concatenate(top_class)
    elif args.task == 'stage':
        top_class = []
        for tc in (
        'Neu_c5-GSTP1(high)OASL(low)', 'Neu_c3-CST7', 'Macro_c3-EREG', 'Epi-Squamous', 'Mega', 'Epi-Ciliated',
        'Mono_c5-CD16', 'Neu_c4-RSAD2', 'Epi-Secretory', 'Neu_c1-IL1B', 'DC_c1-CLEC9A', 'DC_c3-LAMP3',
        'Neu_c2-CXCR4(low)', 'Mono_c4-CD14-CD16', 'Mono_c1-CD14-CCL3', 'Macro_c6-VCAN'):
            top_class.append(indices[cell_type_large == tc])
        selected = np.concatenate(top_class)
    for i in p_ids:
        idx = indices[patient_id == i]
        if len(idx) < 500:
            continue
        if len(set(labels_[idx])) > 1:
            for ii in sorted(set(labels_[idx])):
                if ii > -1:
                    iidx = idx[labels_[idx] == ii]
                    tt_idx = iidx
                    # tt_idx = np.intersect1d(iidx, selected)
                    # tt_idx = np.setdiff1d(iidx, selected)
                    if len(tt_idx) < 1:
                        continue
                    p_idx.append(tt_idx)
                    l_dict[labels_[iidx[0]]] = l_dict.get(labels_[iidx[0]], 0) + 1
        else:
            if args.task == 'severity':
                if (labels_[idx[0]] > -1) and (labels_stage[idx[0]]) > 0:
                    tt_idx = idx
                    # tt_idx = np.intersect1d(idx, selected)
                    # tt_idx = np.setdiff1d(idx, selected)
                    if len(tt_idx) < 1:
                        continue
                    p_idx.append(tt_idx)
                    l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1
            else:
                if labels_[idx[0]] > -1:
                    tt_idx = idx
                    # tt_idx = np.intersect1d(idx, selected)
                    # tt_idx = np.setdiff1d(idx, selected)
                    if len(tt_idx) < 1:
                        continue
                    p_idx.append(tt_idx)
                    l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1

    # print(l_dict)

    return p_idx, labels_, cell_type, patient_id, origin, cell_type_large

# dataloader.py에 있는 Custom_data()를 h5ad가 아니라 
# 이미 load된 AnnData 객체를 받아 처리하는 버전으로 하나 추가하면 됩니다.
def Custom_data_from_loaded(data, args):
    # 1. 라벨 매핑 정의
    id_dict = {
        'normal': 0,
        'COVID-19': 1,
        'hypertrophic cardiomyopathy':1,
        'dilated cardiomyopathy':2,

        'Healthy_stone_donor':0,
        'Healthy_living_donor':0,
        'CKD':1,
        'AKI':2
    }

    # 2. 환자 ID, 라벨, 셀 타입 정보 추출
    patient_id = data.obs['patient'] if 'patient' in data.obs else data.obs['donor_id']
    labels = data.obs['disease__ontology_label']  if 'disease__ontology_label' in data.obs else data.obs['disease_category']

    # cell_type = data.obs[args.cell_type_annotation]
    # 🔧 여기 수정: args.cell_type_annotation 우선 사용
    anno_col = getattr(args, "cell_type_annotation", "manual_annotation")
    if anno_col in data.obs:
        cell_type = data.obs[anno_col].astype("string").fillna("Unknown")
        print("cell type : ", cell_type)
    else:
        print(f"⚠️ '{anno_col}' 컬럼이 없어 'manual_annotation'로 대체합니다.")
        cell_type = data.obs['manual_annotation'].astype("string").fillna("Unknown")
    
    # cell_type = data.obs['manual_annotation']
    # cell_type = data.obs['singler_annotation']

    pd.set_option('display.max_seq_items', None)  # 유니크 항목 출력 제한 해제
    print("cell type annotation : ",cell_type)
    print("✅ [DEBUG] manual_annotation 유니크값:", list(cell_type.unique()))
    # print("✅ [DEBUG] manual_annotation isna sum:", cell_type.isna().sum())
    # print("✅ [DEBUG] manual_annotation dtype:", cell_type.dtype)

    # print("✅ [DEBUG] NaN 위치들:")
    # print(cell_type[cell_type.isna()])

    # 3. expression 데이터 선택
    if args.pca:
        origin = data.obsm['X_pca']
    else:
        origin = data.X.toarray() if not isinstance(data.X, np.ndarray) else data.X

    # 4. 라벨을 숫자로 변환
    labels_ = np.array(labels.map(id_dict))

    # 5. 환자별 인덱스를 구성
    l_dict = {}
    indices = np.arange(origin.shape[0])
    p_ids = sorted(set(patient_id))
    p_idx = []
    
    
    # 환자 단위로 셀을 모아,
    # 다라벨이면 라벨별로 쪼개어 인덱스 묶음을 만들고,
    # 단일 라벨이면 환자 전체 셀 묶음을 만들어,
    # 이 묶음(=후속 단계에서 bag으로 쓰일 원천 집합) 들을 p_idx 리스트에 차곡차곡 쌓는 로직입니다.

    for i in p_ids: # 모든 환자 ID(p_ids)를 하나씩 순회합니다. 여기서 반복 변수 i는 “현재 환자 ID”입니다.
        idx = indices[patient_id == i] # 현재 환자 i에 속하는 셀들의 전체 인덱스를 뽑습니다.
                                        # patient_id == i가 불리언 마스크(길이 = 전체 셀 수)를 만들고,
                                        # 그 마스크로 indices를 필터링해 해당 환자의 셀 인덱스 배열 idx를 얻습니다.
        if len(set(labels_[idx])) > 1:   # one patient with more than one labels # 이 환자 i의 셀들(idx)에 서로 다른 라벨이 2개 이상 있는지 확인합니다.
            for ii in sorted(set(labels_[idx])): # 환자 i에서 라벨별로 나누어 처리하기 위해, 유일 라벨들을 정렬해서 하나씩 순회합니다. # 예: 라벨 집합이 {0, 1, -1}이라면 -1, 0, 1 순으로 돌아요.
                if ii > -1: # 유효 라벨만 사용합니다. # ex: 0,1,2
                    iidx = idx[labels_[idx] == ii] # 현재 라벨 ii에 해당하는 부분 셀 인덱스 묶음을 만듭니다.
                                                    # labels_[idx] == ii는 길이 len(idx)인 불리언 마스크,
                                                    # 그걸로 idx를 다시 필터링하면 “환자 i & 라벨 ii” 조건을 만족하는 셀 인덱스 배열 iidx가 됩니다.
                    tt_idx = iidx
                    # ★ 개수 체크(최소 셀 수 조건) 없이 전부 추가하기 위해 이 코드는 주석처리 하자
                    # if len(tt_idx) < 500:  # exclude the sample with the number of cells fewer than 500
                    #     continue
                    p_idx.append(tt_idx) # 라벨별로 쪼갠 인덱스 묶음(iidx) 을 p_idx 리스트에 넣습니다.
                                            # 나중 단계(샘플링/로더/모델)에서 이 묶음을 “한 bag의 원천 재료”로 사용합니다.
                                            # 다라벨 환자는 라벨 개수만큼 여러 묶음이 생깁니다.
                    l_dict[labels_[iidx[0]]] = l_dict.get(labels_[iidx[0]], 0) + 1     # p_idx에 하나의 인덱스 묶음(= 환자-라벨 그룹)을 추가할 때, 그 묶음의 라벨을 key로 해서 l_dict 값을 +1 증가

        else: # 이 분기는 단일 라벨 환자인 경우(= 유일 라벨 개수 == 1)입니다.
            if labels_[idx[0]] > -1: # 그 단일 라벨이 유효한지 확인합니다. (여기서도 -1은 제외)
                                    # 주의: idx가 비어있다면 idx[0]에서 에러가 납니다. 일반적으로 환자에 최소 1개 셀이 있다고 가정합니다.
                tt_idx = idx
                # ★ 개수 체크(최소 셀 수 조건) 없이 전부 추가하기 위해 이 코드는 주석처리 하자
                # if len(tt_idx) < 500:  # exclude the sample with the number of cells fewer than 500
                #     continue
                p_idx.append(tt_idx) # 단일 라벨 환자는 환자 전체 셀 인덱스 묶음(idx)을 그대로 p_idx에 추가합니다.
                                     # 이렇게 하면, 단일 라벨 환자는 묶음 1개, 다라벨 환자는 라벨 수만큼 여러 묶음이 들어가게 됩니다.
                l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1     # p_idx에 하나의 인덱스 묶음(= 환자-라벨 그룹)을 추가할 때, 그 묶음의 라벨을 key로 해서 l_dict 값을 +1 증가

    print("라벨별 그룹 개수",l_dict) # 다라벨 환자는 라벨마다 여러 묶음으로 잡히니 그만큼 여러 번 카운트됩니다.
    # ex) 라벨별 그룹 개수 {1: 3, 2: 2, 0: 4} # https://chatgpt.com/s/t_689d6521e6088191bb414a646b947ee5 

    # 6. numpy 기반으로 반환
    return p_idx, labels_, np.array(cell_type), np.array(patient_id), origin


# def Custom_data(args):
#     '''
#     !!! Need to change line 178 before running the code !!!
#     '''
#     data = scanpy.read_h5ad(args.dataset)
#     ### Cardio data 실행 코드
#     if args.task == 'custom_cardio':
#         id_dict = {
#             'normal': 0,
#             'hypertrophic cardiomyopathy': 1,
#             'dilated cardiomyopathy': 2
#         }
#         patient_id = data.obs['patient']
#         labels = data.obs['disease__ontology_label']
#         cell_type = data.obs['cell_type_annotation']
    
#     ### Covid data 실행 코드
#     elif args.task == 'custom_covid':
#         id_dict = {
#             'normal': 0,
#             'COVID-19': 1
#         }
#         patient_id = data.obs['donor_id']
#         labels = data.obs['disease__ontology_label']        
#         cell_type = data.obs['cell_type_annotation']
    
#     else:
#         raise ValueError(f"Unsupported task for Custom_data: {args.task}")

#     # data = scanpy.read_h5ad(args.dataset)
#     if args.pca == True:
#         origin = data.obsm['X_pca']
#     else:
#         # origin = data.layers['raw']
#         # pca False ; 수정 2
#         origin = data.X.toarray() if not isinstance(data.X, np.ndarray) else data.X

    
#     # patient_id = data.obs['patient_id']

#     # labels = data.obs['Outcome']

#     # cell_type = data.obs['cell_type']

#     cell_type_large = None
#     # This (high resolution) cell_type is only for attention analysis, not necessary
#     # cell_type_large = data.obs['cell_type_large']

#     labels_ = np.array(labels.map(id_dict))

#     l_dict = {}
#     indices = np.arange(origin.shape[0])
#     p_ids = sorted(set(patient_id))
#     p_idx = []

#     for i in p_ids:
#         idx = indices[patient_id == i]
#         if len(set(labels_[idx])) > 1:   # one patient with more than one labels
#             for ii in sorted(set(labels_[idx])):
#                 if ii > -1:
#                     iidx = idx[labels_[idx] == ii]
#                     tt_idx = iidx
#                     # if len(tt_idx) < 500:  # exclude the sample with the number of cells fewer than 500
#                     if len(tt_idx) < max(args.train_sample_cells, args.test_sample_cells):
#                         continue
#                     p_idx.append(tt_idx)
#                     l_dict[labels_[iidx[0]]] = l_dict.get(labels_[iidx[0]], 0) + 1
#         else:
#             if labels_[idx[0]] > -1:
#                 tt_idx = idx
#                 # if len(tt_idx) < 500:  # exclude the sample with the number of cells fewer than 500
#                 if len(tt_idx) < max(args.train_sample_cells, args.test_sample_cells):
#                     continue
#                 p_idx.append(tt_idx)
#                 l_dict[labels_[idx[0]]] = l_dict.get(labels_[idx[0]], 0) + 1

#     # print(l_dict)

#     return p_idx, labels_, cell_type, patient_id, origin, cell_type_large
