# 0. 설치 (Colab 셀 최상단)
# !pip install --upgrade tensorflow tensorflow-addons

# 1. 라이브러리
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# 2. 설정
base_data_dir = '/content/data_by_title'  # train_by_dept, val_by_dept 폴더가 이 아래에 있어야 함
img_size      = (224,224)
input_shape   = img_size + (3,)
batch_size    = 128
style_dim     = 256
epochs        = 300

# Dept 목록 (폴더명)
depts = [
    'American Paintings and Sculpture',
    'Drawings and Prints',
    'European Paintings',
    'Robert Lehman Collection'
]

# 3. SAFF 레이어 정의
class SAFF(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.fc = Dense(channels, activation='sigmoid')
    def call(self, style_vec, features):
        att = self.fc(style_vec)
        att = tf.reshape(att, (-1,1,1,features.shape[-1]))
        return features + att * features


# 4. 배치 올 트리플렛 로스 (placeholder: 실제 구현을 여기에 import or 정의)
def pairwise_distances(emb):
    """emb: [B, D] -> pairwise squared l2 distance matrix [B, B]"""
    dot = tf.matmul(emb, emb, transpose_b=True)
    sq = tf.reduce_sum(tf.square(emb), axis=1, keepdims=True)
    return tf.maximum(sq - 2.0*dot + tf.transpose(sq), 0.0)

def batch_all_triplet_loss(labels, embeddings, margin=0.2):
    """
    Batch-all triplet loss:
      sum_{i,j,k: y_i=y_j, y_i!=y_k} [d(i,j) - d(i,k) + margin]_+
    normalized by number of valid triplets
    """
    # 1) pairwise distance matrix
    pdist = pairwise_distances(embeddings)  # [B, B]

    # 2) boolean masks
    labels = tf.reshape(labels, (-1,1))
    mask_pos = tf.equal(labels, tf.transpose(labels))  # same class
    mask_neg = tf.not_equal(labels, tf.transpose(labels))  # diff class

    # 3) compute all triplet losses
    #   for each anchor i, for each positive j, for each negative k:
    #     loss = d(i,j) - d(i,k) + margin
    #   we can broadcast to get shape [B, B, B]
    anchor_pos = tf.expand_dims(pdist, 2)   # [B, B, 1]
    anchor_neg = tf.expand_dims(pdist, 1)   # [B, 1, B]
    triplet_loss = anchor_pos - anchor_neg + margin  # [B, B, B]

    # 4) apply masks
    mask = tf.cast(mask_pos, tf.float32)[:,:,None] * tf.cast(mask_neg, tf.float32)[:,None,:]
    triplet_loss = tf.maximum(triplet_loss * mask, 0.0)

    # 5) count non-zero losses
    valid_triplets = tf.cast(triplet_loss > 1e-16, tf.float32)
    num_positive = tf.reduce_sum(valid_triplets)
    # 6) sum and normalize
    loss = tf.reduce_sum(triplet_loss) / (num_positive + 1e-16)
    return loss


# 5. 모델 빌드 함수: 인자를 받도록 수정
def build_pretrained_style_model(input_shape, num_classes, style_dim):
    base = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = True

    feat = base.output
    style_vec = GlobalAveragePooling2D(name='style_gap')(feat)
    style_vec = Dense(style_dim, activation='relu', name='style_fc')(style_vec)

    fused = SAFF(channels=feat.shape[-1], name='saff')(style_vec, feat)

    x = GlobalAveragePooling2D(name='final_gap')(fused)
    preds = Dense(num_classes, activation='softmax', name='predictions')(x)

    return Model(inputs=base.input,
                 outputs=[style_vec, preds],
                 name='EfficientStyleModel')

# 6. 멀티태스크 제너레이터: 두 출력을 위한 y 반환
def multitask_generator(flow, style_dim):
    while True:
        x, y_onehot = next(flow)
        y_int = np.argmax(y_onehot, axis=1)
        # style_fc 레이어 이름과 일치시키세요
        zero_style = np.zeros((x.shape[0], style_dim), dtype=np.float32)
        yield x, {'style_fc': zero_style, 'predictions': y_onehot}

# 7. 전체 루프: Dept별로 모델 생성·학습·저장
os.makedirs('/content/results1', exist_ok=True)

for dept in depts:
    K.clear_session()
    print(f"\n===== Training titles for {dept} =====")
    # 데이터 경로
    train_dir = os.path.join(base_data_dir, 'train_by_dept', dept)
    val_dir   = os.path.join(base_data_dir, 'val_by_dept',   dept)

    # ImageDataGenerator & flow
    train_gen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.8,1.2),
        horizontal_flip=True
    ).flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_gen = ImageDataGenerator(rescale=1/255).flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_titles = train_gen.num_classes
    print(f"Found {train_gen.samples} images across {num_titles} titles")

    # 모델 생성 & 컴파일
    model = build_pretrained_style_model(input_shape, num_titles, style_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            'style_fc':      batch_all_triplet_loss,
            'predictions':   'categorical_crossentropy'
        },
        loss_weights={'style_fc':1.0, 'predictions':1.0},
        metrics={'predictions':'accuracy'}
    )

    # 멀티태스크 제너레이터 래핑
    train_mt = multitask_generator(train_gen, style_dim)
    val_mt   = multitask_generator(val_gen,   style_dim)

    # steps_per_epoch = train_loader.samples // batch_size
    # validation_steps = val_loader.samples // batch_size
    steps_per_epoch = max(1, train_gen.samples // batch_size)
    validation_steps = max(1, val_gen.samples // batch_size)
    print(steps_per_epoch, validation_steps)


    # 콜백
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

    # 학습
    history = model.fit(
        train_mt,
        validation_data=val_mt,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 시각화 저장
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(history.history['loss'], label='train loss')
    axes[0].plot(history.history['val_loss'], '--', label='val loss')
    axes[0].legend(); axes[0].set_title('Loss')

    axes[1].plot(history.history['predictions_accuracy'], label='train acc')
    axes[1].plot(history.history['val_predictions_accuracy'], '--', label='val acc')
    axes[1].legend(); axes[1].set_title('Accuracy')

    save_prefix = dept.replace(' ','_')
    plt.savefig(f"/content/results1/{save_prefix}_metrics.png")
    plt.show() # 추가함. error나면 삭제.
    plt.close(fig)

    # 모델 및 히스토리 저장
    model.save(f"/content/results1/title_model_{save_prefix}.h5")
    pd.DataFrame(history.history).to_csv(
        f"/content/results1/history_{save_prefix}.csv", index=False
    )

    print(f"→ Saved model & history for {dept}")

print("\nAll Dept Title classifiers trained and saved under /content/results1/")
