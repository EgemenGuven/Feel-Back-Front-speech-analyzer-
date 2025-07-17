import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pickle

SAMPLE_RATE = 22050
MAX_LEN = 130

# === Etiket çıkarım fonksiyonları ===
def label_from_ravdess(filename):
    return {
        "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }.get(filename.split("-")[2])

def label_from_emodb(filename):
    return {
        "W": "angry", "L": "neutral", "E": "disgust", "A": "fearful",
        "F": "happy", "T": "sad", "N": "neutral"
    }.get(filename[5].upper())

def label_from_tess(filename):
    name = filename.lower()
    if "angry" in name: return "angry"
    if "disgust" in name: return "disgust"
    if "fear" in name: return "fearful"
    if "happy" in name: return "happy"
    if "neutral" in name: return "neutral"
    if "ps" in name: return "surprised"
    if "sad" in name: return "sad"

def label_from_savee(filename):
    f = filename.lower()
    if "sa" in f: return "sad"
    if "su" in f: return "surprised"
    if "a" in f: return "angry"
    if "d" in f: return "disgust"
    if "f" in f: return "fearful"
    if "h" in f: return "happy"
    if "n" in f: return "neutral"

def label_from_cremad(filename):
    parts = filename.split('_')
    if len(parts) < 3: return None
    return {
        "NEU": "neutral", "HAP": "happy", "SAD": "sad",
        "ANG": "angry", "FEA": "fearful", "DIS": "disgust"
    }.get(parts[2].split('.')[0].upper())

def label_from_jlcorpus(filename):
    name = filename.lower()
    if "angry" in name:
        return "angry"
    elif "anxious" in name or "concerned" in name:
        return "fearful"
    elif "apologetic" in name:
        return "sad"
    elif "assertive" in name:
        return "neutral"
    elif "encouraging" in name or "excited" in name or "happy" in name:
        return "happy"
    elif "neutral" in name:
        return "neutral"
    elif "sad" in name:
        return "sad"
    else:
        return None

def label_from_turev(path):
    # Artık doğrudan dataset_turev klasöründe olduğu için klasör adına göre etiketle
    emotion_dir = os.path.basename(os.path.dirname(path)).lower()

    if emotion_dir == "dataset_turev":
        # Alternatif olarak dosya adı ile belirle
        fname = os.path.basename(path).lower()
        if "angry" in fname or "_acik" in fname:
            return "angry"
        elif "happy" in fname:
            return "happy"
        elif "sad" in fname:
            return "sad"
        elif "calm" in fname:
            return "neutral"
    return None

# === Özellik çıkarımı ===
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = librosa.util.normalize(mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    if combined.shape[1] < MAX_LEN:
        combined = np.pad(combined, pad_width=((0, 0), (0, MAX_LEN - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :MAX_LEN]
    return combined

# === Veri kümeleri tanımı ===
datasets = [
    ("dataset_ravdess", label_from_ravdess),
    ("dataset_emodb", label_from_emodb),
    ("dataset_tess", label_from_tess),
    ("dataset_savee", label_from_savee),
    ("dataset_cremad/AudioWAV", label_from_cremad),
    ("dataset_jlcorpus/JL", label_from_jlcorpus),
    ("dataset_turev", label_from_turev)  # Güncellendi
]

X, y = [], []

for folder, label_func in datasets:
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".wav"):
                label = label_func(os.path.join(root, file))
                if label:
                    try:
                        features = extract_features(os.path.join(root, file))
                        X.append(features[..., np.newaxis])
                        y.append(label)
                        print(f"✔ {file} → {label}")
                    except Exception as e:
                        print(f"⚠️ {file}: {e}")

# === Encode işlemi ===
X = np.array(X)
le = LabelEncoder()
y_encoded = tf.keras.utils.to_categorical(le.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === CNN + BiLSTM ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X.shape[1:]),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Permute((2, 1, 3)),
    tf.keras.layers.Reshape((-1, 32)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# === Değerlendirme ve kayıt ===
y_pred = model.predict(X_test)
y_pred_cls = np.argmax(y_pred, axis=1)
y_true_cls = np.argmax(y_test, axis=1)

print("\n📊 Sınıflandırma Raporu:\n")
print(classification_report(y_true_cls, y_pred_cls, target_names=le.classes_))

f1 = f1_score(y_true_cls, y_pred_cls, average='weighted')
print(f"🎯 Weighted F1 Skoru: {f1:.4f}")

model.save("emotion_model.keras")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Eğitim tamamlandı. Model ve encoder kaydedildi.")