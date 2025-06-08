# cnn_split_augmentation_data.py (수정본)
import os
import json
import re
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import albumentations as A

# -----------------------------
# 설정: 경로 및 파라미터
# -----------------------------
json_file_path = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\labels_with_image_paths.json"
train_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\train_229"
val_dir   = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\val_229"
test_dir  = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\test_229"

TARGET_SIZE = (229, 229)
NUM_AUG = 20        # Train용으로 생성할 강한 증강본 수
VAL_COUNT = 4       # Val용 약한 증강본 수
TEST_COUNT = 4      # Test용 약한 증강본 수

# -----------------------------
# 헬퍼 함수
# -----------------------------
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sanitize_title(title, max_len=50):
    safe = re.sub(r'[^a-zA-Z0-9가-힣_]', '_', title)
    return safe[:max_len]

# -----------------------------
# Albumentations 증강 파이프라인 분리
# -----------------------------

# 1) Train용: 강한 증강(조명/반사, 왜곡, 노이즈, 블러, 회전/크롭, 드롭아웃 등을 강하게 조합)
train_augmentor = A.Compose([
    # 1) 미술관 조명 & 반사
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.8),
    A.CLAHE(p=0.7),
    A.RandomSunFlare(p=0.5),
    A.RandomShadow(p=0.5),
    # 2) 디바이스 왜곡
    A.OpticalDistortion(distort_limit=0.1, p=0.7),
    A.Perspective(distortion_scale=0.1, p=0.7),
    A.Downscale(scale_min=0.5, scale_max=1.0, p=0.5),
    A.ISONoise(color_shift=(0.01,0.1), intensity=(0.1,0.7), p=0.5),
    # 3) 촬영 실수 보정
    A.MotionBlur(blur_limit=(3,9), p=0.5),
    A.GaussianBlur(blur_limit=(3,9), p=0.5),
    A.Rotate(limit=25, p=0.7),
    A.RandomCrop(height=120, width=120, p=0.6),
    # 4) 정규화 기반
    A.CoarseDropout(max_holes=2, max_height=50, max_width=50, p=0.6),
], p=1.0)

# 2) Val/Test용: 비교적 약한 증강(가벼운 밝기/대비, 소량의 블러, 작은 회전 정도만 적용)
val_test_augmentor = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.CLAHE(p=0.3),
    # 디바이스 왜곡은 최소화
    A.OpticalDistortion(distort_limit=0.03, p=0.3),
    A.Perspective(distortion_scale=0.03, p=0.3),
    # 블러/노이즈도 살짝만
    A.MotionBlur(blur_limit=(3,5), p=0.3),
    A.GaussianBlur(blur_limit=(3,5), p=0.3),
    # 회전은 ±10도 이내
    A.Rotate(limit=10, p=0.5),
    # 크롭은 거의 하지 않음(고정된 크기 사용)
    # A.RandomCrop(height=140, width=140, p=0.3),
    # 작은 드롭아웃
    A.CoarseDropout(max_holes=1, max_height=30, max_width=30, p=0.3),
], p=1.0)

# -----------------------------
# 메인 로직: 이미지 증강 및 저장
# -----------------------------
if __name__ == '__main__':
    # 저장 디렉토리 생성
    create_dir(train_dir)
    create_dir(val_dir)
    create_dir(test_dir)

    # JSON 로드
    base_dir = os.path.dirname(json_file_path)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        title = sanitize_title(item['title'])
        rel_path = item['file_path']  # JSON에 기록된 상대 경로

        # glob을 이용해 실제 이미지 파일 경로 찾기
        search_pattern = os.path.abspath(os.path.join(base_dir, rel_path + '*'))
        matches = glob.glob(search_pattern)
        if not matches:
            print(f"[WARN] 이미지 없음: {search_pattern}")
            continue
        img_path = matches[0]  # 첫 번째 매칭 파일 사용

        # 원본 로드(150×150로 리사이즈)
        img = load_img(img_path, target_size=TARGET_SIZE)
        img_arr = img_to_array(img).astype(np.uint8)

        # 클래스 폴더 경로 생성
        train_cls = os.path.join(train_dir, title)
        val_cls   = os.path.join(val_dir, title)
        test_cls  = os.path.join(test_dir, title)

        create_dir(train_cls)
        create_dir(val_cls)
        create_dir(test_cls)

        # 1) 원본 이미지는 Train에 그대로 저장
        array_to_img(img_arr).save(os.path.join(train_cls, f"{title}_orig.jpg"))

        # =============================
        # 2) Train용 강한 증강 생성 & 저장
        # =============================
        for idx in range(1, NUM_AUG + 1):
            aug = train_augmentor(image=img_arr)['image']
            # Augment 후에도 한번 더 150×150 리사이즈 (Crop 등으로 크기가 바뀔 수 있음)
            aug_resized = A.Resize(*TARGET_SIZE)(image=aug)['image']
            array_to_img(aug_resized).save(
                os.path.join(train_cls, f"{title}_train_{idx}.jpg")
            )

        # ===================================
        # 3) Val용 약한 증강 생성 & 저장 (2장)
        # ===================================
        for idx in range(1, VAL_COUNT + 1):
            aug_val = val_test_augmentor(image=img_arr)['image']
            aug_val_resized = A.Resize(*TARGET_SIZE)(image=aug_val)['image']
            array_to_img(aug_val_resized).save(
                os.path.join(val_cls, f"{title}_val_{idx}.jpg")
            )

        # ====================================
        # 4) Test용 약한 증강 생성 & 저장 (2장)
        # ====================================
        for idx in range(1, TEST_COUNT + 1):
            aug_test = val_test_augmentor(image=img_arr)['image']
            aug_test_resized = A.Resize(*TARGET_SIZE)(image=aug_test)['image']
            array_to_img(aug_test_resized).save(
                os.path.join(test_cls, f"{title}_test_{idx}.jpg")
            )

        print(f"[OK] {title}: Train(orig+{NUM_AUG}), Val({VAL_COUNT}), Test({TEST_COUNT}) 이미지 저장 완료.")

    print("=== 모든 클래스 증강 완료 ===")
