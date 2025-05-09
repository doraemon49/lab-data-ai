# Corrected Experiment 2 code with proper freezing logic

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# ====== 1. 데이터 불러오기 및 증강 (기본) ======
img_size = (299, 299)
batch_size = 32
train_dir = '/content/cnn_train_data'
val_dir = '/content/cnn_validation_data'

basic_datagen = ImageDataGenerator(rescale=1./255)
train_gen = basic_datagen.flow_from_directory(train_dir, target_size=img_size,
                                             batch_size=batch_size, class_mode='categorical')
val_gen = basic_datagen.flow_from_directory(val_dir, target_size=img_size,
                                           batch_size=batch_size, class_mode='categorical', shuffle=False)

# ====== 2. freeze 전략 정의 ======
def apply_freeze(base_model, strategy):
    if strategy == 'top_only':
        base_model.trainable = False
    elif strategy == 'unfreeze_last2_blocks':
        # freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False
        # then unfreeze last two Inception blocks: mixed9 and mixed10
        for layer in base_model.layers:
            if layer.name.startswith('mixed9') or layer.name.startswith('mixed10'):
                layer.trainable = True
    elif strategy == 'full_unfreeze':
        base_model.trainable = True

# ====== 3. 하이퍼파라미터 그리드 ======
freeze_strategies = ['top_only', 'unfreeze_last2_blocks', 'full_unfreeze']
lrs = [1e-5, 1e-4, 1e-3]

results = []

# ====== 4. 실험 루프 ======
for strategy in freeze_strategies:
    for lr in lrs:
        tf.keras.backend.clear_session()
        
        # 4.1. 모델 정의
        base = InceptionV3(weights='imagenet', include_top=False,
                           input_shape=(*img_size, 3))
        apply_freeze(base, strategy)
        
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(train_gen.num_classes, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
        )
        
        # 4.2. 콜백
        chkpt_path = f'best_{strategy}_lr{lr:.0e}.h5'
        chkpt = ModelCheckpoint(filepath=chkpt_path, monitor='val_accuracy',
                                save_best_only=True, verbose=1)
        early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        
        # 4.3. 학습
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=20,
            callbacks=[chkpt, early],
            verbose=1
        )
        
        # 4.4. 평가
        loss, acc, auc = model.evaluate(val_gen, verbose=0)
        results.append({
            'strategy': strategy,
            'learning_rate': lr,
            'accuracy': acc,
            'auc': auc,
            'epochs_trained': len(history.history['loss'])
        })
        print(f"{strategy}, lr={lr}: acc={acc:.4f}, auc={auc:.4f}")

# ====== 5. 결과 저장 및 시각화 ======
df = pd.DataFrame(results)
df.to_csv('experiment2_results.csv', index=False)
print("\nExperiment 2 results:\n", df)

# heatmap of accuracy
pivot_acc = df.pivot(index='strategy', columns='learning_rate', values='accuracy')
plt.figure(figsize=(6, 4))
sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="viridis")
plt.title('Val Accuracy by Freeze Strategy & LR')
plt.savefig('exp2_accuracy_heatmap.png')

