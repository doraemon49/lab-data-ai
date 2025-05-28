# title_subclass_with_test_evaluation.py
# 0. 설치 (Colab 셀 최상단)
# !pip install --upgrade tensorflow tensorflow-addons

# 1. 라이브러리
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import pandas as pd
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 2. 설정
base_data_dir = '/content/data_by_class'  # train_by_dept, val_by_dept, test_by_dept 폴더가 이 아래에 있음
img_size      = (224,224)
input_shape   = img_size + (3,)
batch_size    = 128
style_dim     = 256
epochs        = 300

depts = [
    'American Paintings and Sculpture',
    'Drawings and Prints',
    'European Paintings',
    'Robert Lehman Collection'
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
    print(f"\n===== Training titles for {dept} =====")
    train_dir = os.path.join(base_data_dir, 'train', dept)
    val_dir   = os.path.join(base_data_dir, 'val',   dept)

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
    model.summary()
    param_count = model.count_params()
    print(f"📦 Total parameters: {param_count:,}")

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
    axes[0].plot(history.history['predictions_loss'], label='train loss')
    axes[0].plot(history.history['val_predictions_loss'], '--', label='val loss')
    axes[0].legend(); axes[0].set_title('Loss')
    axes[1].plot(history.history['predictions_accuracy'], label='train acc')
    axes[1].plot(history.history['val_predictions_accuracy'], '--', label='val acc')
    axes[1].legend(); axes[1].set_title('Accuracy')

    prefix = dept.replace(' ', '_')
    plt.savefig(f"/content/results1/{prefix}_metrics.png")
    plt.show()
    plt.close(fig)
    model.save(f"/content/results1/title_model_{prefix}.keras")
    pd.DataFrame(history.history).to_csv(f"/content/results1/history_{prefix}.csv", index=False)

    # 9. 테스트 평가
    print(f"--- Testing titles for {dept} ---")
    test_dir = os.path.join(base_data_dir, 'test', dept)
    test_flow = ImageDataGenerator(rescale=1/255).flow_from_directory(
        test_dir, target_size=img_size,
        batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    test_mt = multitask_generator(test_flow)
    test_steps = int(np.ceil(test_flow.samples / batch_size))

    # 9-1. 모델 정량 평가
    test_res = model.evaluate(test_mt, steps=test_steps, verbose=1, return_dict=True)
    print(f"Test results for {dept}:")
    for name, value in test_res.items():
        print(f"{name}: {value:.4f}")

    # 9-2. 예측 결과 추출
    predict_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('predictions').output)
    pred_probs = predict_model.predict(test_flow, steps=test_steps, verbose=1)
    y_pred = np.argmax(pred_probs, axis=1)
    y_true = test_flow.classes[:len(y_pred)]

    # 9-3. class 이름 및 class 수
    class_names = list(test_flow.class_indices.keys())
    num_classes = len(class_names)

    # 9-4. Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n🔍 Classification Report:\n", report)

    # 9-5. Top-3 Accuracy 계산
    from tensorflow.keras.utils import to_categorical
    y_true_onehot = to_categorical(y_true, num_classes=num_classes)
    top3 = np.mean(tf.keras.metrics.top_k_categorical_accuracy(y_true_onehot, pred_probs, k=3).numpy())
    print(f"🔝 Top-3 Accuracy: {top3:.4f}")

    # 9-6. Confusion Matrix 시각화
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {dept}')
    plt.tight_layout()
    plt.savefig(f'/content/results1/confmat_{prefix}.png')
    plt.show()
    plt.close()

    # 9-7. Style 벡터 추출 및 t-SNE 시각화
    style_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('style_fc').output)
    style_vecs = style_model.predict(test_flow, steps=test_steps, verbose=1)

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(style_vecs)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(tsne_result[:,0], tsne_result[:,1], c=y_true, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(num_classes), label='True Label')
    plt.title(f't-SNE of Style Vectors - {dept}')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig(f'/content/results1/tsne_style_vecs_{prefix}.png')
    plt.show()
    plt.close()

    # 9-8. 파일 저장
    save_path = f"/content/results1/test_results_{prefix}.txt"
    with open(save_path, 'w') as f:
        for name, value in test_res.items():
            f.write(f"{name}: {value:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"→ Saved test results, confusion matrix, and t-SNE to /content/results1/ for {dept}")

print("\nAll Dept Title classifiers trained, tested, and saved under /content/results1/")