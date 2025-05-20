import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import ttest_rel
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

# ===== 1. Data Generators =====
img_size   = (224, 224)
batch_size = 64

train_dir = '/content/train_data'
val_dir   = '/content/validation_data'
test_dir  = '/content/test_data'             # ← test data 경로 추가

datagen   = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_gen   = datagen.flow_from_directory(val_dir,   target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
test_gen  = datagen.flow_from_directory(test_dir,  target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)  # ← 추가

num_classes = train_gen.num_classes

# ===== 2. Build Base + Head =====
def build_model(base_fn, input_shape):
    base = base_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

models = {
    'VGG16': (VGG16, (224,224,3)),
    'InceptionV3': (InceptionV3, (299,299,3)),
    'ResNet50': (ResNet50, (224,224,3)),
    'EfficientNetB0': (EfficientNetB0, (224,224,3)),
    'DenseNet121': (DenseNet121, (224,224,3)),
    'MobileNetV2': (MobileNetV2, (224,224,3)),
    'Xception':    (Xception,    (299,299,3)),
    'EfficientNetV2B0': (EfficientNetV2B0, (224,224,3))
}

results = {}
histories = {}
all_test_accuracies = {}   # ← 테스트 정확도 저장용

# ===== 3. Training & Evaluation Loop =====
for name, (fn, shape) in models.items():
    print(f"=== {name} ===")
    # generator target size 조정
    train_gen = datagen.flow_from_directory(train_dir, target_size=shape[:2], batch_size=batch_size, class_mode='categorical')
    val_gen   = datagen.flow_from_directory(val_dir,   target_size=shape[:2], batch_size=batch_size, class_mode='categorical', shuffle=False)
    test_gen  = datagen.flow_from_directory(test_dir,  target_size=shape[:2], batch_size=batch_size, class_mode='categorical', shuffle=False)

    model = build_model(fn, shape)
    chkpt = ModelCheckpoint(f'best_{name}.h5', monitor='val_accuracy', save_best_only=True)
    early = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # ── 학습
    history = model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=[chkpt, early], verbose=2)
    histories[name] = history.history

    # ── Validation 예측
    val_gen.reset()
    y_val_pred       = model.predict(val_gen, verbose=0)
    y_val_true       = val_gen.classes
    y_val_pred_label = np.argmax(y_val_pred, axis=1)
    val_acc  = accuracy_score(y_val_true, y_val_pred_label)
    val_prec = precision_score(y_val_true, y_val_pred_label, average='macro')
    val_rec  = recall_score(y_val_true, y_val_pred_label, average='macro')
    val_f1   = f1_score(y_val_true, y_val_pred_label, average='macro')
    val_cm   = confusion_matrix(y_val_true, y_val_pred_label)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # ── Test 예측 (추가)
    test_gen.reset()
    y_test_pred       = model.predict(test_gen, verbose=0)
    y_test_true       = test_gen.classes
    y_test_pred_label = np.argmax(y_test_pred, axis=1)
    test_acc  = accuracy_score(y_test_true, y_test_pred_label)
    test_prec = precision_score(y_test_true, y_test_pred_label, average='macro')
    test_rec  = recall_score(y_test_true, y_test_pred_label, average='macro')
    test_f1   = f1_score(y_test_true, y_test_pred_label, average='macro')
    test_cm   = confusion_matrix(y_test_true, y_test_pred_label)
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")

    # ── 혼동 행렬 저장
    plt.figure(figsize=(6,6))
    plt.imshow(test_cm, cmap='Blues')
    plt.title(f'{name} Test Confusion Matrix')
    plt.colorbar()
    plt.savefig(f'{name}_test_confusion_matrix.png')
    plt.close()

    # ── 결과 저장
    results[name] = {
        'val_acc' : val_acc,  'val_prec' : val_prec,  'val_rec' : val_rec,  'val_f1' : val_f1,
        'test_acc': test_acc, 'test_prec': test_prec, 'test_rec': test_rec, 'test_f1': test_f1,
        'total_params': model.count_params()
    }
    all_test_accuracies[name] = test_acc
"""
=== VGG16 ===
val_acc: 0.7150
val_prec: 0.7481
val_rec: 0.7150
val_f1: 0.6993
test_acc: 0.6959
test_prec: 0.7257
test_rec: 0.6959
test_f1: 0.6777
total_params: 15932026

=== InceptionV3 ===
val_acc: 0.2739
val_prec: 0.2501
val_rec: 0.2739
val_f1: 0.2402
test_acc: 0.2404
test_prec: 0.2356
test_rec: 0.2404
test_f1: 0.2201
total_params: 24592986

=== ResNet50 ===
val_acc: 0.3519
val_prec: 0.3513
val_rec: 0.3519
val_f1: 0.3224
test_acc: 0.3217
test_prec: 0.3405
test_rec: 0.3217
test_f1: 0.3023
total_params: 26377914

=== EfficientNetB0 ===
val_acc: 0.0032
val_prec: 0.0000
val_rec: 0.0032
val_f1: 0.0000
test_acc: 0.0032
test_prec: 0.0000
test_rec: 0.0032
test_f1: 0.0000
total_params: 6053341

=== DenseNet121 ===
val_acc: 0.2054
val_prec: 0.2041
val_rec: 0.2054
val_f1: 0.1875
test_acc: 0.1911
test_prec: 0.1924
test_rec: 0.1911
test_f1: 0.1750
total_params: 8779130

=== MobileNetV2 ===
val_acc: 0.2341
val_prec: 0.2349
val_rec: 0.2341
val_f1: 0.2137
test_acc: 0.2341
test_prec: 0.2337
test_rec: 0.2341
test_f1: 0.2150
total_params: 4261754

=== Xception ===
val_acc: 0.0048
val_prec: 0.0005
val_rec: 0.0048
val_f1: 0.0009
test_acc: 0.0048
test_prec: 0.0004
test_rec: 0.0048
test_f1: 0.0007
total_params: 23651682

=== EfficientNetV2B0 ===
val_acc: 0.0064
val_prec: 0.0001
val_rec: 0.0064
val_f1: 0.0001
test_acc: 0.0064
test_prec: 0.0001
test_rec: 0.0064
test_f1: 0.0001
total_params: 7923082
"""