import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tqdm import tqdm

# ===== 사용자 설정 =====
input_dir = '/content/cnn_train_data'     # 원본 학습 데이터 폴더
output_dir = '/content/cnn_train_aug'     # 증강된 이미지 저장 폴더
augmentations_per_image = 3               # 이미지 1장당 몇 개 생성할지

# ===== 증강기 설정 =====
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ===== 증강 함수 =====
def augment_and_save(class_name, img_path, save_dir):
    img = load_img(img_path)                  # PIL image
    x = img_to_array(img)                     # (H, W, C)
    x = np.expand_dims(x, axis=0)             # (1, H, W, C)

    # 저장 경로 준비
    basename = os.path.splitext(os.path.basename(img_path))[0]

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir,
                              save_prefix=basename, save_format='jpeg'):
        i += 1
        if i >= augmentations_per_image:
            break  # 지정된 개수만큼 저장

# ===== 메인 루프 =====
os.makedirs(output_dir, exist_ok=True)
class_names = os.listdir(input_dir)

for class_name in tqdm(class_names, desc="클래스별 증강 중"):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    image_files = os.listdir(class_input_path)
    for img_file in image_files:
        img_path = os.path.join(class_input_path, img_file)
        augment_and_save(class_name, img_path, class_output_path)

import shutil
import os
from tqdm import tqdm

orig_dir = '/content/cnn_train_data'
aug_dir = '/content/cnn_train_aug'
merged_dir = '/content/cnn_train_all'

os.makedirs(merged_dir, exist_ok=True)

for class_name in os.listdir(orig_dir):
    orig_class_path = os.path.join(orig_dir, class_name)
    aug_class_path  = os.path.join(aug_dir,  class_name)
    merged_class_path = os.path.join(merged_dir, class_name)
    os.makedirs(merged_class_path, exist_ok=True)

    # 원본 복사
    for fname in os.listdir(orig_class_path):
        shutil.copy(os.path.join(orig_class_path, fname), os.path.join(merged_class_path, fname))

    # 증강 복사
    for fname in os.listdir(aug_class_path):
        shutil.copy(os.path.join(aug_class_path, fname), os.path.join(merged_class_path, fname))
