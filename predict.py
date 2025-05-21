import librosa
import numpy as np
import joblib

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

def predict_emotion(file_path):
    model = joblib.load("emotion_model.pkl")
    le = joblib.load("label_encoder.pkl")

    features = extract_features(file_path)
    pred_encoded = model.predict(features)[0]
    emotion = le.inverse_transform([pred_encoded])[0]

    return emotion

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    emotion = predict_emotion(audio_file)
    print(f"Predicted Emotion: {emotion}")
