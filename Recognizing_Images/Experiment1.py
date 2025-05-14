import tensorflow as tf
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import matplotlib.pyplot as plt

# ====== 1. 데이터 불러오기 및 증강 ======
batch_size = 32
train_dir = '/content/cnn_train_data'
val_dir = '/content/cnn_validation_data'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# ====== 2. 모델별 설정 ======
model_configs = {
    'VGG16': {
        'model_fn': VGG16,
        'input_shape': (224, 224, 3),
        'lr': 1e-4,
        'head': lambda x, num_classes: Dense(num_classes, activation='softmax')(Dense(512, activation='relu')(x))
    },
    'InceptionV3': {
        'model_fn': InceptionV3,
        'input_shape': (299, 299, 3),
        'lr': 1e-4,
        'head': lambda x, num_classes: Dense(num_classes, activation='softmax')(Dense(512, activation='relu')(x))
    },
    'ResNet50': {
        'model_fn': ResNet50,
        'input_shape': (224, 224, 3),
        'lr': 5e-5,
        'head': lambda x, num_classes: Dense(num_classes, activation='softmax')(Dropout(0.3)(Dense(256, activation='relu')(x)))
    },
    'EfficientNetB0': {
        'model_fn': EfficientNetB0,
        'input_shape': (240, 240, 3),
        'lr': 1e-5,
        'head': lambda x, num_classes: Dense(num_classes, activation='softmax')(Dropout(0.3)(Dense(128, activation='relu')(x)))
    },
}

results = {}
histories = {}

# ====== 3. 각 모델 학습 ======
for model_name, config in model_configs.items():
    print(f"\n▶ Training: {model_name}")

    input_shape = config['input_shape']
    lr = config['lr']
    model_fn = config['model_fn']

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=input_shape[:2], batch_size=batch_size, class_mode='categorical', shuffle=False
    )

    base_model = model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # freeze base

    x = GlobalAveragePooling2D()(base_model.output)
    output = config['head'](x, train_gen.num_classes)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=f'best_model_{model_name.lower()}.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks
    )

    histories[model_name] = history.history
    loss, acc, auc = model.evaluate(val_gen)
    results[model_name] = {'accuracy': acc, 'auc': auc}

# ====== 4. 결과 요약 ======
print("\n✅ 결과 요약:")
for k, v in results.items():
    print(f"{k}: Accuracy={v['accuracy']:.4f}, AUC={v['auc']:.4f}")

# ====== 5. 학습 시각화 ======
for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
    plt.figure()
    for name, hist in histories.items():
        if metric in hist:
            plt.plot(hist[metric], label=name)
    plt.title(f'{metric.upper()} Comparison')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric}_comparison.png')
    plt.close()

for model_name, hist in histories.items():
    for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
        if metric in hist:
            plt.figure()
            plt.plot(hist[metric])
            plt.title(f'{model_name} - {metric.upper()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.grid(True)
            plt.savefig(f'{model_name.lower()}_{metric}.png')
            plt.close()
