import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, GlobalAveragePooling2D,
                                     Dense, Concatenate, BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os

# 1. 데이터 제너레이터
img_size = (299, 299)
batch_size = 32
train_dir = '/content/cnn_train_data'
val_dir = '/content/cnn_validation_data'

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(train_dir, target_size=img_size,
                                        batch_size=batch_size, class_mode='categorical')
val_gen = datagen.flow_from_directory(val_dir, target_size=img_size,
                                      batch_size=batch_size, class_mode='categorical', shuffle=False)
num_classes = train_gen.num_classes

# 2. 베이스 모델 (InceptionV3 freeze)
base = InceptionV3(weights='imagenet', include_top=False,
                   input_shape=(*img_size, 3))
base.trainable = False
features = base.output

# 3. 멀티-브랜치 정의
# 색감 branch
c = Conv2D(64, (1,1), padding='same')(features)
c = BatchNormalization()(c)
c = Activation('relu')(c)
c = GlobalAveragePooling2D()(c)

# 형태 branch
s1 = Conv2D(64, (3,3), padding='same')(features)
s2 = Conv2D(64, (5,5), padding='same')(features)
s = Concatenate()([s1, s2])
s = BatchNormalization()(s)
s = Activation('relu')(s)
s = GlobalAveragePooling2D()(s)

# 질감 branch
t = Conv2D(64, (7,7), padding='same')(features)
t = BatchNormalization()(t)
t = Activation('relu')(t)
t = GlobalAveragePooling2D()(t)

# 4. 분기 합치고 헤드
merged = Concatenate()([c, s, t])
merged = Dense(512, activation='relu')(merged)
outputs = Dense(num_classes, activation='softmax')(merged)

model = Model(inputs=base.input, outputs=outputs)

# 5. 컴파일 및 콜백
model.compile(optimizer=Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)])
os.makedirs('models', exist_ok=True)
chkpt = ModelCheckpoint('models/best_multibranch.h5', monitor='val_accuracy',
                        save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# 6. 학습
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=30, callbacks=[chkpt, early], verbose=1)

# 7. 평가 및 저장
loss, acc, auc = model.evaluate(val_gen, verbose=0)
pd.DataFrame([{'val_accuracy': acc, 'val_auc': auc, 'val_loss': loss}]) \
  .to_csv('multibranch_results.csv', index=False)
print(f"Multibranch Model - Val Acc: {acc:.4f}, Val AUC: {auc:.4f}, Val Loss: {loss:.4f}")
