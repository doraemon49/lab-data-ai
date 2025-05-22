# title_subclass_with_test_evaluation.py
# 0. 설치 (Colab 셀 최상단)
# !pip install --upgrade tensorflow tensorflow-addons

# 1. 라이브러리
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.regularizers import l2


# 2. 설정
base_data_dir = '/content/'  # train_by_dept, val_by_dept, test_by_dept 폴더가 이 아래에 있음
img_size      = (224,224)
input_shape   = img_size + (3,)
batch_size    = 128
style_dim     = 256
epochs        = 300

depts = [
    'American Paintings and Sculpture'#,
    # 'Drawings and Prints',
    # 'European Paintings',
    # 'Robert Lehman Collection'
]

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

# 5. 모델 빌드 함수
def build_pretrained_style_model(input_shape, num_classes, style_dim):
    base = InceptionResNetV2(
        weights='imagenet', include_top=False, input_shape=input_shape
    )
    base.trainable = True
    feat = base.output
    style_vec = GlobalAveragePooling2D(name='style_gap')(feat)
    style_vec = Dense(style_dim, activation='relu', name='style_fc')(style_vec)
    fused = SAFF(channels=feat.shape[-1], name='saff')(style_vec, feat)
    x = GlobalAveragePooling2D(name='final_gap')(fused)
    x = Dropout(0.5, name='dropout')(x)  ########## ← 추가
    preds = Dense(num_classes, activation='softmax',kernel_regularizer=l2(1e-4),  name='predictions')(x) ########## ← L2 규제 추가
    return Model(inputs=base.input, outputs=[style_vec, preds], name='EfficientStyleModel')

# 6. 멀티태스크 제너레이터 (style labels로 수정)
def multitask_generator(flow):
    while True:
        x, y_onehot = next(flow)
        y_int = np.argmax(y_onehot, axis=1).astype(np.int32)
        yield x, {'style_fc': y_int, 'predictions': y_onehot}

# 7. 학습 및 평가 루프
os.makedirs('/content/results1', exist_ok=True)

for dept in depts:
    K.clear_session()
    print(f"\n===== Training titles for {dept} =====")
    train_dir = os.path.join(base_data_dir, 'train_by_dept', dept)
    val_dir   = os.path.join(base_data_dir, 'val_by_dept',   dept)

    train_gen = ImageDataGenerator(
        rescale=1/255, rotation_range=20,
        width_shift_range=0.1, height_shift_range=0.1,
        brightness_range=(0.8,1.2), horizontal_flip=True
    ).flow_from_directory(
        train_dir, target_size=img_size,
        batch_size=batch_size, class_mode='categorical'
    )
    val_gen = ImageDataGenerator(rescale=1/255).flow_from_directory(
        val_dir, target_size=img_size,
        batch_size=batch_size, class_mode='categorical'
    )

    num_titles = train_gen.num_classes
    print(f"Found {train_gen.samples} images and {num_titles} titles in {dept}")

    model = build_pretrained_style_model(input_shape, num_titles, style_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={'style_fc': batch_all_triplet_loss, 'predictions': 'categorical_crossentropy'},
        loss_weights={'style_fc':0.5, 'predictions':1.0},  #### ← style weight 줄여보기
        metrics={'predictions':'accuracy'}
    )

    train_mt = multitask_generator(train_gen)
    val_mt   = multitask_generator(val_gen)
    steps_per_epoch = max(1, train_gen.samples // batch_size)
    validation_steps = max(1, val_gen.samples // batch_size)

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_mt, validation_data=val_mt,
        steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        epochs=epochs, callbacks=callbacks, verbose=1
    )

    # 8. 결과 저장 및 시각화
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(history.history['loss'], label='train loss')
    axes[0].plot(history.history['val_loss'], '--', label='val loss')
    axes[0].legend(); axes[0].set_title('Loss')
    axes[1].plot(history.history['predictions_accuracy'], label='train acc')
    axes[1].plot(history.history['val_predictions_accuracy'], '--', label='val acc')
    axes[1].legend(); axes[1].set_title('Accuracy')

    prefix = dept.replace(' ', '_')
    plt.savefig(f"/content/results1/{prefix}_metrics.png")
    plt.close(fig)
    model.save(f"/content/results1/title_model_{prefix}.h5")
    pd.DataFrame(history.history).to_csv(f"/content/results1/history_{prefix}.csv", index=False)

    # 9. 테스트 평가
    print(f"--- Testing titles for {dept} ---")
    test_dir = os.path.join(base_data_dir, 'test_by_dept', dept)
    test_flow = ImageDataGenerator(rescale=1/255).flow_from_directory(
        test_dir, target_size=img_size,
        batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    test_mt = multitask_generator(test_flow)
    test_steps = max(1, test_flow.samples // batch_size)

    # return_dict=True 로 dict 반환
    test_res = model.evaluate(test_mt, steps=test_steps, verbose=1, return_dict=True)

    print(f"Test results for {dept}:")
    for name, value in test_res.items():
        print(f"{name}: {value:.4f}")

    # accuracy만 별도 추출
    test_acc = test_res.get('predictions_accuracy', test_res.get('accuracy'))

    # 파일 저장
    save_path = f"/content/results1/test_results_{prefix}.txt"
    with open(save_path, 'w') as f:
        for name, value in test_res.items():
            f.write(f"{name}: {value:.4f}\n")
        f.write(f"test_accuracy: {test_acc:.4f}\n")

    print(f"→ Saved test results (including accuracy) to {save_path}")


print("\nAll Dept Title classifiers trained, tested, and saved under /content/results1/")