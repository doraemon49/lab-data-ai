import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# 1) test_loader에서 파일 경로 리스트 만들기
# -------------------------------------------------
if hasattr(test_loader, 'filepaths'):
    # tf.keras >=2.7 DirectoryIterator
    filepaths = test_loader.filepaths
elif hasattr(test_loader, 'filenames') and hasattr(test_loader, 'directory'):
    # tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory
    filepaths = [os.path.join(test_loader.directory, fname)
                 for fname in test_loader.filenames]
else:
    raise AttributeError("test_loader에 filepaths나 filenames+directory 속성이 없습니다.")

# -------------------------------------------------
# 2) 쿼리 이미지 선택 및 스타일 벡터
# -------------------------------------------------
query_index = 100       # ▶ 원하는 인덱스로 바꿔보세요 #6 사람, 27, 32, (38),47
# (50, 74, 79, 87)
query_path  = filepaths[query_index]
query_vec   = style_vecs[query_index]

# -------------------------------------------------
# 3) 거리/유사도 계산
# -------------------------------------------------
metric = 'euclidean'         # ▶ 'euclidean' 또는 'cosine'
if metric == 'euclidean':
    # 전체 벡터와의 유클리드 거리
    dists = np.linalg.norm(style_vecs - query_vec, axis=1)
    # 거리 작은 순으로 정렬
    sorted_idx = np.argsort(dists)
elif metric == 'cosine':
    # 전체 벡터와의 코사인 유사도
    sims = cosine_similarity(query_vec.reshape(1, -1), style_vecs)[0]
    # 유사도 큰 순으로 정렬
    sorted_idx = np.argsort(-sims)
else:
    raise ValueError("metric을 'euclidean' 또는 'cosine' 중 하나로 설정하세요.")

# 자기 자신(쿼리) 제외 후 Top-4
sorted_idx = sorted_idx[sorted_idx != query_index]
top5_idx   = sorted_idx[:4]

# -------------------------------------------------
# 4) 시각화
# -------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(18, 6))

# — Query 이미지
img = Image.open(query_path)
axes[0].imshow(img)
axes[0].set_title('Query')
axes[0].axis('off')

# — Top-4 Neighbors
for rank, idx in enumerate(top5_idx, start=1):
    img = Image.open(filepaths[idx])
    axes[rank].imshow(img)
    if metric == 'euclidean':
        txt = f'Rank {rank}\nDist: {dists[idx]:.3f}'
    else:
        txt = f'Rank {rank}\nSim: {sims[idx]:.3f}'
    axes[rank].set_title(txt)
    axes[rank].axis('off')

plt.suptitle('Top-4 Nearest Neighbors by ' +
             ('Euclidean Distance' if metric=='euclidean' else 'Cosine Similarity'),
             fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('/content/results3/nn_style_search.png')
plt.show()
