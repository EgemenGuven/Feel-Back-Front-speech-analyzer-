# type: ignore
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tensorflow as tf
import pickle
from collections import Counter

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

SAMPLE_RATE = 22050
SEGMENT_DURATION = 5  # saniye

# Model ve encoder yükle
model = tf.keras.models.load_model("emotion_model.keras")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc = librosa.util.normalize(mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    max_len = 130
    if combined.shape[1] < max_len:
        combined = np.pad(combined, pad_width=((0, 0), (0, max_len - combined.shape[1])), mode='constant')
    else:
        combined = combined[:, :max_len]
    return combined[..., np.newaxis]

@app.route("/api/predict-emotion", methods=["POST"])
def predict_single():
    if "audio" not in request.files:
        return jsonify({"error": "Ses dosyası gönderilmedi"}), 400

    file = request.files["audio"]
    try:
        y, _ = librosa.load(file, sr=SAMPLE_RATE)
        features = extract_mfcc(y)
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features, verbose=0)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction))
        return jsonify({
            "emotion": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict-segments", methods=["POST"])
def predict_segments():
    if "audio" not in request.files:
        return jsonify({"error": "Ses dosyası gönderilmedi"}), 400

    file = request.files["audio"]
    try:
        y, sr = librosa.load(file, sr=SAMPLE_RATE)
        segment_samples = SEGMENT_DURATION * sr
        total_segments = int(np.ceil(len(y) / segment_samples))
        predictions = []
        time_segments = []

        for i in range(total_segments):
            start = i * segment_samples
            end = min((i + 1) * segment_samples, len(y))
            segment = y[start:end]
            if len(segment) < sr * 1:
                continue
            features = extract_mfcc(segment)
            features = np.expand_dims(features, axis=0)
            pred = model.predict(features, verbose=0)
            label = label_encoder.inverse_transform([np.argmax(pred)])[0]
            predictions.append(label)
            time_segments.append({
                "start": round(start / sr, 2),
                "end": round(end / sr, 2),
                "emotion": label
            })

        counts = dict(Counter(predictions))
        total = sum(counts.values())
        percentage_distribution = {emo: round(100 * count / total, 2) for emo, count in counts.items()}

        return jsonify({
            "emotion_distribution": percentage_distribution,
            "segments": time_segments
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
