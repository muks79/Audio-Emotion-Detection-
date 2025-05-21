# 🎧 Emotion-Based Playlist Generator using Voice Input

This project is an intelligent music recommendation system that detects the user's **emotion from voice input** and plays a **matching playlist** based on the mood. It uses audio processing and machine learning to classify emotions from recorded speech and recommend songs accordingly.

---

## 🚀 Features

- 🎤 **Voice Input-Based Emotion Detection**
- 🧠 **Trained from Scratch** using your voice recordings (no pretrained model)
- 🎼 **Emotion-Based Playlist Generator** (happy, sad, angry, etc.)
- 💾 **Model Export** using `joblib` for later predictions
- 📊 **Real-time Emotion Prediction** from .wav audio files

---

## 🛠️ Technologies Used

- **Python**
- **librosa** – Audio feature extraction (MFCC)
- **scikit-learn** – Model training (Random Forest)
- **joblib** – Saving the model
- **pandas & numpy** – Data handling
- **os, soundfile** – File I/O
- (Optional) **pygame/vlc** – For playing music automatically

---

## 📁 Folder Structure

```
📦emotion-playlist-generator
 ┣ 📂 recordings/             # Your voice recordings (.wav)
 ┣ 📂 playlists/              # Song folders by emotion
 ┣ 📜 emotion_model.pkl       # Trained ML model
 ┣ 📜 train.py                # Trains the model
 ┣ 📜 predict_and_play.py     # Predicts emotion & plays playlist
 ┣ 📜 utils.py                # Helper code
 ┗ 📜 README.md               # Project documentation
```

---

## 🔧 How It Works

1. 🎙️ **User records voice** expressing an emotion.
2. 📊 **MFCC features** are extracted from the voice using `librosa`.
3. 🌲 A **Random Forest classifier** is trained on your labeled voice samples.
4. 🧠 The model **predicts emotion** from a new audio input.
5. 🎵 The system **plays a song** from the playlist that matches the detected emotion.

---

## ▶️ Getting Started

### 1. Install Dependencies
```bash
pip install numpy pandas librosa scikit-learn joblib soundfile
```

### 2. Record Audio Samples
Place `.wav` files labeled with emotions (e.g., `happy_1.wav`, `sad_2.wav`) in the `recordings/` folder.

### 3. Train the Model
```bash
python train.py
```

### 4. Predict Emotion and Play Playlist
```bash
python predict_and_play.py your_audio_file.wav
```

---

## 📊 Supported Emotions

- Happy  
- Sad  
- Angry  
- Neutral

> You can extend this list by adding more labeled voice samples and playlists.

---

## 💡 What You’ll Learn

- How to extract meaningful features from audio
- How to build a supervised ML pipeline from scratch
- Hands-on with `scikit-learn`, `librosa`, and `joblib`
- Real-world integration of ML with media automation

---

## 🔮 Future Enhancements

- Add deep learning support (CNN/LSTM)
- Add a web UI for recording and recommendations
- Improve emotion detection accuracy with larger datasets

---

## 👨‍💻 Author

**Mukul Mehra**  
📧 mukulmehra681@gmail.com  
🔗 GitHub: [muks79](https://github.com/muks79)

