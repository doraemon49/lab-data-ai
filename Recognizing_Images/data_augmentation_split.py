# $ python models/two_cnn/data/cnn_split_augmentation_data.py
import os
import json
import re
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import albumentations as A

# -----------------------------
# 설정: 경로 및 파라미터
# -----------------------------
json_file_path = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\labels_with_image_paths.json"
train_dir = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_train_data"
val_dir   = r"C:\Users\LG\Documents\MJU\Activity\SKT_FLY_AI\github\AI\app\models\two_cnn\data\cnn_validation_data"
TARGET_SIZE = (150, 150)
NUM_AUG = 10
VAL_RATIO = 0.2

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
# Albumentations 증강 파이프라인
# -----------------------------
augmentor = A.Compose([
    # 1) 미술관 조명 & 반사
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.CLAHE(p=0.5),
    A.RandomSunFlare(p=0.3),
    A.RandomShadow(p=0.3),
    # 2) 디바이스 왜곡
    A.OpticalDistortion(distort_limit=0.05, p=0.5),
    A.Perspective(distortion_scale=0.05, p=0.5),
    A.Downscale(scale_min=0.7, scale_max=1.0, p=0.4),
    A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.5), p=0.4),
    # 3) 촬영 실수 보정
    A.MotionBlur(blur_limit=(3,7), p=0.4),
    A.GaussianBlur(blur_limit=(3,7), p=0.4),
    A.Rotate(limit=15, p=0.5),
    A.RandomCrop(height=140, width=140, p=0.5),
    # 4) 정규화 기반
    A.CoarseDropout(max_holes=1, max_height=40, max_width=40, p=0.5),
], p=1.0)

# -----------------------------
# 메인 로직: 이미지 증강 및 저장
# -----------------------------
if __name__ == '__main__':
    create_dir(train_dir)
    create_dir(val_dir)

    base_dir = os.path.dirname(json_file_path)
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        title = sanitize_title(item['title'])
        rel_path = item['file_path']  # JSON의 기초 경로

        # glob을 이용해 실제 파일 경로 검색
        search_pattern = os.path.abspath(os.path.join(base_dir, rel_path + '*'))
        matches = glob.glob(search_pattern)
        if not matches:
            print(f"[WARN] 이미지 없음: {search_pattern}")
            continue
        img_path = matches[0]  # 첫 번째 매칭 파일 사용

        # 원본 로드
        img = load_img(img_path, target_size=TARGET_SIZE)
        img_arr = img_to_array(img).astype(np.uint8)

        # 저장 경로 생성
        train_cls = os.path.join(train_dir, title)
        val_cls   = os.path.join(val_dir, title)
        create_dir(train_cls)
        create_dir(val_cls)

        # 원본 저장
        array_to_img(img_arr).save(os.path.join(train_cls, f"{title}_orig.jpg"))

        # 증강 이미지 생성
        aug_list = []
        for _ in range(NUM_AUG):
            aug = augmentor(image=img_arr)['image']
            aug_resized = A.Resize(*TARGET_SIZE)(image=aug)['image']
            aug_list.append(aug_resized)

        # train/val 분할 및 저장
        train_imgs, val_imgs = train_test_split(aug_list, test_size=VAL_RATIO, random_state=42)
        for idx, arr in enumerate(train_imgs, 1):
            array_to_img(arr).save(os.path.join(train_cls, f"{title}_train_{idx}.jpg"))
        for idx, arr in enumerate(val_imgs, 1):
            array_to_img(arr).save(os.path.join(val_cls, f"{title}_val_{idx}.jpg"))

        print(f"[OK] {title}: train={len(train_imgs)}, val={len(val_imgs)} 이미지 저장 완료.")

    print("=== 모든 클래스 증강 완료 ===")
