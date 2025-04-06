import pickle
import pandas as pd

# 파일 경로 목록 (세포 타입 관련 파일)
cell_type_files = {
    "COMBAT_cell_type_large": "data/COMBAT/cell_type_large.pkl",
    "COMBAT_cell_type": "data/COMBAT/cell_type.pkl",
    "Haniffa_cell_type_large": "data/Haniffa/cell_type_large.pkl",
    "Haniffa_cell_type": "data/Haniffa/cell_type.pkl",
}

# 데이터 로드 및 출력
for key, file in cell_type_files.items():
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            print(f"📂 {file} 데이터 타입: {type(data)}")

            # pandas.Series인 경우 개수 및 고유값 출력
            if isinstance(data, pd.Series):
                print(f"  📊 샘플 개수: {data.shape[0]}")
                print(f"  🔍 예제 데이터: {data.head(5).to_dict()}")

                # 고유한 세포 타입 확인
                unique_values = data.unique()
                num_unique_values = len(unique_values)
                print(f"  🏷️ Unique 세포 타입 개수: {num_unique_values}")
                print(f"  🔹 Unique Cell Types: {unique_values}")

            # list나 dict인 경우 개수 출력
            elif isinstance(data, list) or isinstance(data, dict):
                print(f"  📊 샘플 개수: {len(data)}")
                print(f"  🔍 예제 데이터: {data[:5] if isinstance(data, list) else list(data.items())[:5]}")

    except FileNotFoundError:
        print(f"⚠️ 파일을 찾을 수 없음: {file}")

    print("-" * 50)
