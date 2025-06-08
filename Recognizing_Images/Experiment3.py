import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

# ====== 1. 데이터 로드 ======
img_size = (299, 299)
batch_size = 32
train_dir = '/content/cnn_train_data'
val_dir   = '/content/cnn_validation_data'

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
val_gen = datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)

num_classes = train_gen.num_classes

# ====== 2. 공통 설정 ======
BASE_LR = 1e-4
FREEZE_STRATEGY = 'top_only'  # InceptionV3 전체 freeze

# ====== 3. 헤드 구조 정의 ======
def build_head(base_output, head_type):
    if head_type == 'simple_fc':
        x = Flatten()(base_output)
        out = Dense(num_classes, activation='softmax')(x)

    elif head_type == 'deep_fc':
        x = Flatten()(base_output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(num_classes, activation='softmax')(x)

    elif head_type == 'gap':
        x = GlobalAveragePooling2D()(base_output)
        out = Dense(num_classes, activation='softmax')(x)

    else:
        raise ValueError(f"Unknown head type: {head_type}")
    return out

head_types = ['simple_fc', 'deep_fc', 'gap']

# ====== 4. 실험 루프 ======
results = []
for head in head_types:
    tf.keras.backend.clear_session()

    # 4.1 Base model 정의 & Freeze
    base = InceptionV3(weights='imagenet', include_top=False,
                       input_shape=(*img_size, 3))
    base.trainable = False  # top_only

    # 4.2 Head 붙이기
    outputs = build_head(base.output, head)
    model = Model(inputs=base.input, outputs=outputs)

    # 4.3 컴파일
    model.compile(
        optimizer=Adam(learning_rate=BASE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )

    # 4.4 콜백
    chkpt = ModelCheckpoint(
        filepath=f'best_head_{head}.h5',
        monitor='val_accuracy', save_best_only=True, verbose=1
    )
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # 4.5 학습
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=[chkpt, early],
        verbose=1
    )

    # 4.6 평가 및 파라미터·수렴 기록
    loss, acc, auc = model.evaluate(val_gen, verbose=0)
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)

    results.append({
        'head': head,
        'val_accuracy': acc,
        'val_auc': auc,
        'val_loss': loss,
        'epochs': len(history.history['loss']),
        'total_params': total_params,
        'trainable_params': int(trainable_params),
    })

    print(f"[{head}] Val Acc={acc:.4f}, Val AUC={auc:.4f}, Val Loss={loss:.4f}")

# ====== 5. 결과 정리 ======
df = pd.DataFrame(results)
df.to_csv('experiment3_head_results.csv', index=False)
print("\nExperiment 3 (head structures) results:\n", df)

print("Test")
print("아무리 수정해도, git이 추적하지 않음.")
print("staging이후")