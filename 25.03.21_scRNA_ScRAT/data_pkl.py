import pickle
import pandas as pd

# 파일 경로 목록
file_paths = {
    "COMBAT_labels": "data/COMBAT/labels.pkl",
    "COMBAT_patient_id": "data/COMBAT/patient_id.pkl",
    "Haniffa_labels": "data/Haniffa/labels.pkl",
    "Haniffa_patient_id": "data/Haniffa/patient_id.pkl",
}

# 데이터 로드 및 출력
for key, file in file_paths.items():
    with open(file, 'rb') as f:
        data = pickle.load(f)
        print(f"📂 {file} 데이터 타입: {type(data)}")

        # pandas.Series인 경우 개수 및 고유값 출력
        if isinstance(data, pd.Series):
            print(f"  📊 샘플 개수: {data.shape[0]}")
            print(f"  🔍 예제 데이터: {data.head(5).to_dict()}")

            # 고유한 값 확인
            unique_values = data.unique()
            num_unique_values = len(unique_values)
            print(f"  🏷️ Unique 값 개수: {num_unique_values}")

            # 고유한 label 또는 환자 ID 출력
            if "labels" in key:
                print(f"  🔹 Unique Labels: {unique_values}")
                print(f"  📈 Label별 세포 수: {data.value_counts()}")
            elif "patient_id" in key:
                print(f"  🔹 Unique Patient IDs 개수: {num_unique_values}")

        # list나 dict인 경우 개수 출력
        elif isinstance(data, list) or isinstance(data, dict):
            print(f"  📊 샘플 개수: {len(data)}")
            print(f"  🔍 예제 데이터: {data[:5] if isinstance(data, list) else list(data.items())[:5]}")

    print("-" * 50)
