import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Emotions list including new ones
emotions = ['happy', 'sad', 'angry', 'fear', 'surprise']

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def load_data(dataset_path='audio_dataset'):
    X, y = [], []
    for emotion in emotions:
        folder = os.path.join(dataset_path, emotion)
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder, filename)
                features = extract_features(file_path)
                X.append(features)
                y.append(emotion)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_data()

    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Splitting data...")
    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = X, X, y_encoded, y_encoded

    print("Training model...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    print("Testing accuracy...")
    accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("Saving model and label encoder...")
    joblib.dump(model, "emotion_model.pkl")
    joblib.dump(le, "label_encoder.pkl")

    print("Training complete.")
