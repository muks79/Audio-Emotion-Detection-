import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import joblib
import os

st.title("🎧 Emotion-Based Playlist Generator")

model = joblib.load("emotion_model.pkl")
le = joblib.load("label_encoder.pkl")

PLAYLISTS = {
    'happy': [
        ("Happy – Pharrell", "https://www.youtube.com/watch?v=ZbZSe6N_BXs", "😄"),
        ("Good Life – OneRepublic", "https://www.youtube.com/watch?v=jZhQOvvV45w", "🌞")
    ],
    'sad': [
        ("Fix You – Coldplay", "https://www.youtube.com/watch?v=k4V3Mo61fJM", "😢"),
        ("Someone Like You – Adele", "https://www.youtube.com/watch?v=hLQl3WQQoQ0", "💔")
    ],
    'angry': [
        ("Breaking the Habit – Linkin Park", "https://www.youtube.com/watch?v=v2H4l9RpkwM", "😠"),
        ("Stronger – Kanye West", "https://www.youtube.com/watch?v=PsO6ZnUZI0g", "💪")
    ],
    'fear': [
        ("Fear – Imagine Dragons", "https://www.youtube.com/watch?v=Gf1E-IkW6yc", "😨"),
        ("Breathe Me – Sia", "https://www.youtube.com/watch?v=wbP0cTz3Mhg", "😰")
    ],
    'surprise': [
        ("Surprise Yourself – Jack Garratt", "https://www.youtube.com/watch?v=dXY5WnZk2Y0", "😲"),
        ("Alive – Sia", "https://www.youtube.com/watch?v=t2NgsJrrAyM", "⚡")
    ]
}

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

def record_audio(filename="voice.wav", duration=5, fs=44100):
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    st.success("Recording saved!")

if st.button("🎙️ Record Voice"):
    record_audio()

if os.path.exists("voice.wav"):
    features = extract_features("voice.wav")
    pred_encoded = model.predict(features)[0]
    emotion = le.inverse_transform([pred_encoded])[0]

    st.subheader(f"🧠 Detected Emotion: **{emotion.upper()}**")

    st.subheader("🎵 Recommended Playlist:")
    for title, url, emoji in PLAYLISTS.get(emotion, []):
        st.markdown(f"{emoji} [{title}]({url})")
