# Experiment2.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# ===== 1. Data Generators =====
img_size   = (224, 224)
batch_size = 64

train_dir = '/content/train_data'
val_dir   = '/content/validation_data'
test_dir  = '/content/test_data'

datagen   = ImageDataGenerator(rescale=1./255)
def make_gens(target_size):
    train_gen = datagen.flow_from_directory(train_dir, target_size=target_size,
                                            batch_size=batch_size, class_mode='categorical')
    val_gen   = datagen.flow_from_directory(val_dir,   target_size=target_size,
                                            batch_size=batch_size, class_mode='categorical', shuffle=False)
    test_gen  = datagen.flow_from_directory(test_dir,  target_size=target_size,
                                            batch_size=batch_size, class_mode='categorical', shuffle=False)
    return train_gen, val_gen, test_gen

# ===== 2. Freeze 전략 정의 =====
def apply_freeze(base_model, strategy):
    if strategy == 'orig_freeze30':
        # exactly as in Exp1
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
    elif strategy == 'top_only':
        base_model.trainable = False
    elif strategy == 'unfreeze_last2_blocks':
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers:
            if layer.name.startswith('block5_'):
                layer.trainable = True
    elif strategy == 'full_unfreeze':
        base_model.trainable = True
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# ===== 3. 모델 빌드 함수 (freeze + lr 파라미터) =====
def build_model(freeze_strategy, lr):
    # VGG16 불러오기
    base = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size,3))
    apply_freeze(base, freeze_strategy)

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    num_classes = train_gen.num_classes
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ===== 4. 실험 설정 =====
freeze_strategies = [
    'orig_freeze30',       # original Exp1 setting
    'top_only',
    'unfreeze_last2_blocks',
    'full_unfreeze'
]
learning_rates = [1e-5, 1e-4, 1e-3]  # make sure 1e-5 (Exp1’s lr) is included

results = {}

# ===== 5. 실험 루프 =====
for strategy in freeze_strategies:
    for lr in learning_rates:
        exp_name = f"{strategy}_lr{lr}"
        print(f"\n=== Experiment: {exp_name} ===")
        train_gen, val_gen, test_gen = make_gens(img_size)

        # 모델 생성
        model = build_model(strategy, lr)

        # 콜백 정의
        chkpt = ModelCheckpoint(f'best_{exp_name}.h5',
                                monitor='val_accuracy', save_best_only=True)
        early = EarlyStopping(monitor='val_accuracy',
                              patience=10, restore_best_weights=True)

        # 학습
        start = time.time()
        history = model.fit(train_gen,
                            validation_data=val_gen,
                            epochs=100,
                            callbacks=[chkpt, early],
                            verbose=2)
        duration = time.time() - start

        # Validation 평가
        val_gen.reset()
        y_val_pred_labels = np.argmax(model.predict(val_gen), axis=1)
        y_val_true        = val_gen.classes
        val_acc = accuracy_score(y_val_true, y_val_pred_labels)

        # Test 평가
        test_gen.reset()
        y_test_pred_labels = np.argmax(model.predict(test_gen), axis=1)
        y_test_true        = test_gen.classes
        test_acc = accuracy_score(y_test_true, y_test_pred_labels)

        # 결과 저장
        results[exp_name] = {
            'strategy': strategy,
            'lr': lr,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'duration_sec': round(duration, 1)
        }
        print(f"{exp_name} → val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}, time: {duration:.1f}s")

# ===== 6. 결과 요약 출력 =====
print("\n=== All Results ===")
for k, v in results.items():
    print(f"{k}: val_acc={v['val_acc']:.4f}, test_acc={v['test_acc']:.4f}, time={v['duration_sec']}s")
