import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image  # 만약 여전히 image 모듈이 필요하다면
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.metrics import TopKCategoricalAccuracy

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="keras.src.trainers.data_adapters")

# ——————————————
# 1) 커스텀 객체 정의
# ——————————————
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
        cfg = super().get_config()
        cfg.update({'channels': self.channels})
        return cfg

def pairwise_distances(emb):
    dot = tf.matmul(emb, emb, transpose_b=True)
    sq = tf.reduce_sum(tf.square(emb), axis=1, keepdims=True)
    return tf.maximum(sq - 2.0*dot + tf.transpose(sq), 0.0)
def batch_all_triplet_loss(labels, embeddings, margin=0.2):
    labels = tf.reshape(labels, (-1,1))
    pdist = pairwise_distances(embeddings)
    pos = tf.equal(labels, tf.transpose(labels))
    neg = tf.not_equal(labels, tf.transpose(labels))
    ap = tf.expand_dims(pdist,2); an = tf.expand_dims(pdist,1)
    tl = tf.maximum(ap - an + margin, 0.0)
    mask = tf.cast(pos,tf.float32)[:,:,None]*tf.cast(neg,tf.float32)[:,None,:]
    tl = tl*mask
    valid = tf.cast(tl>1e-16,tf.float32)
    num_pos = tf.reduce_sum(valid)
    return tf.reduce_sum(tl)/(num_pos+1e-16)

# ——————————————
# 2) Dept 모델 로드 & 평가
# ——————————————
# dept_model_path = '/content/results/style_saff_department.keras'
# dept_model_path = '/content/drive/MyDrive/SKT_FLY_AI/paper/results/style_saff_department.keras'
# dept_model_path = '/content/drive/MyDrive/Colab_Notebooks/paper_skt/results/style_saff_department.keras'
dept_model_path = '/content/drive/MyDrive/results/style_saff_department.keras'

dept_model = load_model(
    dept_model_path,
    custom_objects={
        'SAFF': SAFF,
        'custom_layers>SAFF': SAFF,
        'batch_all_triplet_loss': batch_all_triplet_loss
    },
    compile=True
)

# 테스트 제너레이터
test_dir = '/content/data_by_class/test'
# test_dir = '/content/test'

img_size = (224,224); batch_size=128
test_gen = ImageDataGenerator(rescale=1/255)
test_loader = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# 7. 멀티태스크 제너레이터
def multitask_generator(flow):
    while True:
        x, y_onehot = next(flow)
        y_int = np.argmax(y_onehot, axis=1)
        yield x, {'style_fc': y_int, 'predictions': y_onehot}

test_mt = multitask_generator(test_loader)
steps = int(np.ceil(test_loader.samples / batch_size))  # 522/128 → 5

# 2.1) evaluate 로 loss, acc
eval_res = dept_model.evaluate(
    test_mt,
    steps=steps,
    verbose=1
)
# eval_res = [total_loss, style_fc_loss, pred_loss, pred_acc]
total_loss, tri_loss, dept_loss, dept_acc = eval_res

# 2.2) classification_report, f1_score, confusion_matrix
dept_preds = dept_model.predict(
    test_loader,
    steps=steps,        # ← 반드시 넣어야 5배치만 돌고 멈춥니다.
    verbose=1
)[1]
y_true_dept = test_loader.classes
y_pred_dept = np.argmax(dept_preds, axis=1)
dept_names  = list(test_loader.class_indices.keys())

# one-hot 인코딩
num_classes = len(test_loader.class_indices)       # = 4
y_true_onehot = tf.keras.utils.to_categorical(
    y_true_dept,
    num_classes=num_classes
)

# f1
dept_f1 = f1_score(y_true_dept, y_pred_dept, average='weighted')
# top-3 acc
top3 = TopKCategoricalAccuracy(k=3)
top3.update_state(y_true_onehot, dept_preds)
dept_top3 = top3.result().numpy()

print(f"Top-3 Accuracy: {dept_top3:.4f}")

print("\n▶ Dept Classification Metrics")
print(f"  total loss       : {total_loss:.4f}")
print(f"  triplet loss     : {tri_loss:.4f}")
print(f"  pred loss        : {dept_loss:.4f}")
print(f"  accuracy         : {dept_acc:.4f}")
print(f"  f1 (weighted)       : {dept_f1:.4f}")
print(f"  top-3 accuracy   : {dept_top3:.4f}")

print("\nDept Classification Report:")
print(classification_report(
    y_true_dept, y_pred_dept, target_names=dept_names
))

cm_dept = confusion_matrix(y_true_dept, y_pred_dept)
plt.figure(figsize=(6,5))
sns.heatmap(cm_dept, annot=True, fmt='d',
            xticklabels=dept_names, yticklabels=dept_names,
            cmap='Blues')
plt.title('Dept Confusion Matrix')
plt.xlabel('Pred'); plt.ylabel('True')
plt.tight_layout()
plt.savefig('/content/results2/Dept_confusion_matrix.png')
plt.show()
plt.close()

# ──────────────────────────────
# (Dept 평가 후) t-SNE 시각화
# ──────────────────────────────
from sklearn.manifold import TSNE

# 1) style_vec(embeddings)만 뽑아오기
#    model.predict(...) 의 첫 번째 출력이 style_vec
steps = int(np.ceil(test_loader.samples / batch_size))  # 522/128 → 5

style_vecs = dept_model.predict(
    test_loader,
    steps=steps,
    verbose=1
)[0]  # [N, style_dim]

# 2) t-SNE 로 2차원으로 축소
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(style_vecs)  # shape (N,2)

# 3) 산점도 그리기
plt.figure(figsize=(8,6))
for idx, name in enumerate(dept_names):
    mask = (y_true_dept == idx)
    plt.scatter(
        emb_2d[mask, 0], emb_2d[mask, 1],
        label=name, alpha=0.7
    )
plt.legend()
plt.title('t-SNE of Dept Embeddings')
plt.xlabel('TSNE-1')
plt.ylabel('TSNE-2')
plt.tight_layout()
plt.savefig('/content/results2/tsne_style_vecs1.png')
plt.show()
plt.close()


# ——————————————
# 3) Title 파이프라인 평가
# ——————————————
depts = [
    'American Paintings and Sculpture',
    'Drawings and Prints',
    'European Paintings',
    'Robert Lehman Collection'
]

# —— 파일명용 안전한 키 생성 함수
def sanitize(dept_name):
    return dept_name.replace(' ', '_').replace('&','and')


# 3.1) Title 모델들 로드 & 클래스맵
depts = dept_names
title_models, title_classes = {}, {}
for d in depts:
    key = d                              # e.g. "Drawings and Prints"
    fname = sanitize(d)                 # "Drawings_and_Prints"
    # path = f"/content/results1/title_model_{fname}.keras"
    # path = f"/content/drive/MyDrive/SKT_FLY_AI/paper/results1/title_model_{fname}.keras"
    # path = f"/content/drive/MyDrive/Colab_Notebooks/paper_skt/results1/title_model_{fname}.keras"
    path = f"/content/drive/MyDrive/results1/title_model_{fname}.keras"

    tm = load_model(path, custom_objects={
        'SAFF': SAFF,
        'custom_layers>SAFF': SAFF,
        'batch_all_triplet_loss': batch_all_triplet_loss
    }, compile=True)
    # compile for loss+top3
    tm.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 TopKCategoricalAccuracy(k=3, name='top_3_acc')]
    )
    title_models[key] = tm
    # 클래스 이름(폴더명)
    title_classes[d] = sorted(os.listdir(os.path.join(test_dir, d)))

# 3.2) 샘플별 예측 루프
# losses, corr1, corr3 = [], 0, 0
# y_true_titles, y_pred_titles = [], []

# for i, fp in enumerate(test_loader.filepaths):
#     # --- 경로에서 “실제” 부서, 타이틀 뽑기
#     true_dp = os.path.basename(os.path.dirname(os.path.dirname(fp)))
#     true_t  = os.path.basename(os.path.dirname(fp))

#     # 이미지 로드
#     img = image.load_img(fp, target_size=img_size)
#     x = image.img_to_array(img)/255.0
#     x = np.expand_dims(x,0)

#     # Dept 예측
#     dp = dept_names[y_pred_dept[i]]

#     # --- 예측 부서가 실제랑 다르면 건너뛰기
#     if dp != true_dp:
#         # (필요하다면 corr1/corr3 카운트에도 반영하세요—일종의 틀린 예측으로 처리)
#         continue

#     # --- 같은 부서일 때만 title_classes[dp] 에서 인덱스 찾기
#     # true title
#     true_t = os.path.basename(os.path.dirname(fp))
#     # title 예측
#     tm = title_models[dp]
#     _, preds_t = tm.predict(x, verbose=0)
#     # loss
#     cls_list = title_classes[dp]
#     idx = cls_list.index(true_t)
#     onehot = tf.keras.utils.to_categorical([idx], num_classes=len(cls_list))
#     l = tf.keras.losses.categorical_crossentropy(onehot, preds_t).numpy()[0]
#     losses.append(l)
#     # top-1/3
#     order = np.argsort(preds_t[0])[::-1]
#     if order[0]==idx: corr1+=1
#     if idx in order[:3]: corr3+=1
#     # for classification_report
#     y_true_titles.append(true_t)
#     y_pred_titles.append(cls_list[order[0]])

# N = len(losses)
# avg_loss = np.mean(losses)
# acc1 = corr1/N
# acc3 = corr3/N
# # title f1 (string labels)
# title_f1 = f1_score(y_true_titles, y_pred_titles, average='weighted')

# print("\n▶ Title Pipeline Metrics")
# print(f"  samples       : {N}")
# print(f"  avg loss      : {avg_loss:.4f}")
# print(f"  top-1 acc     : {acc1:.4f}")
# print(f"  top-3 acc     : {acc3:.4f}")
# print(f"  f1 (weighted)    : {title_f1:.4f}")

# print("\nTitle Classification Report (weighted):")
# print(classification_report(y_true_titles, y_pred_titles))

"""
건너뛰지 않고 모든 522개 샘플에 대해
dept_pred가 true_dp가 아니면 “타이틀도 틀린 것으로” 처리
맞춘 건 “부서도 맞고 타이틀도 맞은 것”만 카운트
"""
from sklearn.metrics import classification_report, f1_score
import numpy as np

# ──── 전체 Title 클래스 리스트(flatten) ────
all_titles = []
for cls_list in title_classes.values():
    all_titles.extend(cls_list)

# ──── 초기화 ────
losses, corr1, corr3 = [], 0, 0
y_true_titles, y_pred_titles = [], []

# ──── 샘플별 루프 ────
for i, fp in enumerate(test_loader.filepaths):
    # 실제 부서·타이틀
    true_dp = os.path.basename(os.path.dirname(os.path.dirname(fp)))
    true_t  = os.path.basename(os.path.dirname(fp))
    y_true_titles.append(true_t)

    # Dept 예측
    dp_pred = dept_names[y_pred_dept[i]]

    if dp_pred == true_dp:
        # 1) 맞춘 부서에 대해서만 title 모델 호출
        tm = title_models[dp_pred]
        img = image.load_img(fp, target_size=img_size)
        x   = np.expand_dims(image.img_to_array(img)/255.0, 0)
        _, preds_t = tm.predict(x, verbose=0)

        cls_list = title_classes[dp_pred]
        order    = np.argsort(preds_t[0])[::-1]
        pred_t   = cls_list[order[0]]

        # Loss 계산
        idx     = cls_list.index(true_t)
        onehot  = tf.keras.utils.to_categorical([idx], num_classes=len(cls_list))
        loss    = tf.keras.losses.categorical_crossentropy(onehot, preds_t).numpy()[0]
        losses.append(loss)

        # Top-1 / Top-3 카운트
        if order[0] == idx:     corr1 += 1
        if idx in order[:3]:    corr3 += 1
    else:
        # 2) 부서 틀리면 title도 틀린 것으로 간주
        pred_t = "__wrong__"
        losses.append(None)

    y_pred_titles.append(pred_t)

# ──── 파이프라인 전체 성능 계산 ────
N = len(test_loader.filepaths)   # 522
pipeline_acc1 = corr1 / N
pipeline_acc3 = corr3 / N
# Loss는 dept 맞춘 경우만 모아서 평균
avg_loss = np.mean([l for l in losses if l is not None])

# F1-score
pipeline_f1 = f1_score(
    y_true_titles,
    y_pred_titles,
    labels=all_titles,
    average='weighted',
    zero_division=0
)

# ──── 결과 출력 ────
print("▶ Pipeline Overall Metrics")
print(f"  samples    : {N}")
print(f"  avg loss   : {avg_loss:.4f}")
print(f"  top-1 acc  : {pipeline_acc1:.4f}")
print(f"  top-3 acc  : {pipeline_acc3:.4f}")
print(f"  f1 (wtd)   : {pipeline_f1:.4f}")

print("\n▶ Full Title Classification Report")
print(classification_report(
    y_true_titles,
    y_pred_titles,
    labels=all_titles,
    zero_division=0
))