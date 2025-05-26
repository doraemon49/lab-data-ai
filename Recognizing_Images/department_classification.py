# 0. 설치 (Colab 셀 최상단)
# !pip install --upgrade tensorflow tensorflow-addons

# 1. 라이브러리
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
# from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow.keras.backend as K
from tensorflow.keras.utils import register_keras_serializable


# 2. 설정
train_dir   = '/content/train'
val_dir     = '/content/val'
test_dir    = '/content/test'
img_size    = (224,224)
input_shape = img_size + (3,)
batch_size  = 128
num_classes = 4
style_dim   = 256
epochs      = 300

# 3. SAFF 레이어 정의 (직렬화 등록)
@register_keras_serializable(package='custom_layers', name='SAFF')
class SAFF(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.fc = Dense(channels, activation='sigmoid')
    def call(self, style_vec, features):
        att = self.fc(style_vec)
        att = tf.reshape(att, (-1,1,1,features.shape[-1]))
        return features + att * features
    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config

# 4. 배치 올 트리플릿 로스
def pairwise_distances(emb):
    dot = tf.matmul(emb, emb, transpose_b=True)
    sq = tf.reduce_sum(tf.square(emb), axis=1, keepdims=True)
    return tf.maximum(sq - 2.0*dot + tf.transpose(sq), 0.0)

def batch_all_triplet_loss(labels, embeddings, margin=0.2):
    # labels: integer class ids, shape [batch]
    labels = tf.reshape(labels, (-1,1))
    pdist = pairwise_distances(embeddings)
    mask_pos = tf.equal(labels, tf.transpose(labels))
    mask_neg = tf.not_equal(labels, tf.transpose(labels))
    anchor_pos = tf.expand_dims(pdist, 2)
    anchor_neg = tf.expand_dims(pdist, 1)
    triplet_loss = anchor_pos - anchor_neg + margin
    mask = tf.cast(mask_pos, tf.float32)[:,:,None] * tf.cast(mask_neg, tf.float32)[:,None,:]
    triplet_loss = tf.maximum(triplet_loss * mask, 0.0)
    valid_triplets = tf.cast(triplet_loss > 1e-16, tf.float32)
    num_positive = tf.reduce_sum(valid_triplets)
    return tf.reduce_sum(triplet_loss) / (num_positive + 1e-16)


# 5. 모델 빌드
def build_pretrained_style_model():
    base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = True

    feat = base.output                                      # [B,7,7,1280]
    style_vec = GlobalAveragePooling2D(name='style_gap')(feat)
    style_vec = Dense(style_dim, activation='relu',
                      name='style_fc')(style_vec)

    fused = SAFF(channels=feat.shape[-1], name='saff')(
        style_vec, feat
    )

    x = GlobalAveragePooling2D(name='final_gap')(fused)
    x = Dropout(0.5, name='dropout')(x)  ########## ← 추가

    preds = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4),
                  name='predictions')(x)

    return Model(inputs=base.input,
                 outputs=[style_vec, preds],
                 name='EfficientStyleModel')

model = build_pretrained_style_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
      'style_fc': batch_all_triplet_loss,
      'predictions': 'categorical_crossentropy'
    },
    loss_weights={'style_fc':0.5, 'predictions':1.0},
    metrics={'predictions':'accuracy'}
)
model.summary()

param_count = model.count_params()
print(f"📦 Total parameters: {param_count:,}")


# 6. 데이터 로더
train_gen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    brightness_range=(0.8,1.2), horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1/255)

train_loader = train_gen.flow_from_directory(
    train_dir, target_size=img_size,
    batch_size=batch_size, class_mode='categorical'
)
val_loader = val_gen.flow_from_directory(
    val_dir, target_size=img_size,
    batch_size=batch_size, class_mode='categorical'
)

# 7. 멀티태스크 제너레이터
def multitask_generator(flow):
    while True:
        x, y_onehot = next(flow)
        y_int = np.argmax(y_onehot, axis=1)
        yield x, {'style_fc': y_int, 'predictions': y_onehot}

train_mt = multitask_generator(train_loader)
val_mt   = multitask_generator(val_loader)
steps_per_epoch = train_loader.samples // batch_size
validation_steps = val_loader.samples // batch_size

# 8. 콜백
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-5, verbose=1
    ),
    EarlyStopping(
        monitor='val_loss', patience=20,
        restore_best_weights=True, verbose=1
    )
]

# 9. 학습
history = model.fit(
    train_mt,
    validation_data=val_mt,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# 10. 시각화, 저장, 평가
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['predictions_loss'], label='train loss')
plt.plot(history.history['val_predictions_loss'], '--', label='val loss')
plt.legend(); plt.title('Loss')

plt.subplot(1,2,2)
# 'accuracy' → 'predictions_accuracy'
plt.plot(history.history['predictions_accuracy'], label='train acc')
plt.plot(history.history['val_predictions_accuracy'], '--', label='val acc')
plt.legend(); plt.title('Accuracy')

plt.savefig('/content/results/metrics.png')
plt.show()

# 11. 모델·히스토리 저장
os.makedirs('/content/results', exist_ok=True)
model.save('/content/results/style_saff_department.h5')
pd.DataFrame(history.history).to_csv(
    '/content/results/history.csv', index=False
)

