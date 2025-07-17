# single_emotion_test.py – WAV dosyasından tek duygu tahmini yapar (TESS + gelişmiş model uyumlu)
import librosa
import numpy as np
import tensorflow as tf
import pickle
import os

SAMPLE_RATE = 22050
MAX_LEN = 130

# === MFCC + delta + delta2 ===
def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc = librosa.util.normalize(mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])  # (39, time)
    if combined.shape[1] < MAX_LEN:
        combined = np.pad(combined, pad_width=((0, 0), (0, MAX_LEN - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :MAX_LEN]
    return combined[..., np.newaxis]

# === Model ve encoder yükle ===
print("📦 Model ve encoder yükleniyor...")
model = tf.keras.models.load_model("emotion_model.keras")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === Tek duygu tahmini fonksiyonu ===
def predict_single_emotion(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    features = extract_mfcc(audio)
    features = np.expand_dims(features, axis=0)  # (1, 39, 130, 1)
    prediction = model.predict(features, verbose=0)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction)
    print(f"\n🎭 Tahmin edilen duygu: {predicted_label} (Güven: {confidence:.2f})")

# === Test Başlat ===
if __name__ == "__main__":
    test_path = input("🗂️ Test edilecek WAV dosyasının yolu: ").strip()
    if os.path.exists(test_path):
        predict_single_emotion(test_path)
    else:
        print("❌ Dosya bulunamadı.")
