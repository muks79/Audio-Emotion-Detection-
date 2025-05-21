# 🎧 Emotion-Based Playlist Generator using Voice Input

This project detects the **emotion** in a user's **voice recording (.wav)** and recommends a **music playlist** that matches the mood. Built using **Python**, **Machine Learning**, and **Streamlit**, it includes custom-trained models and a simple web interface for real-time emotion prediction.

---

## 📁 Project Structure

```
Emotion-Based-Playlist-Generator/
├── audio_dataset/
│   ├── happy/
│   │   ├── happy1.wav, ...
│   ├── sad/
│   │   ├── sad1.wav, ...
│   ├── angry/
│   ├── fear/
│   └── surprise/
├── app.py               # Streamlit UI
├── train.py             # Model training script
├── predict.py           # Emotion prediction logic
├── emotion_model.pkl    # Saved ML model
├── requirements.txt
└── README.md
```

---

## 🧠 Features

- 🎙️ Record/upload `.wav` audio
- 🤖 ML-based emotion detection (happy, sad, angry, fear, surprise)
- 🎵 Recommend a mood-based YouTube playlist
- 😃 Streamlit UI with emojis and visual feedback

---

## 🔧 Requirements

Make sure you have Python installed (>= 3.8). Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Run

### 1. Add Training Data

Add `.wav` files under `audio_dataset/` folder in subfolders like:

- `happy/`, `sad/`, `angry/`, `fear/`, `surprise/`

Ensure each class has **at least 2-3 `.wav` files**.

### 2. Train the Model

```bash
python train.py
```

It creates `emotion_model.pkl`.

### 3. Run the App

```bash
streamlit run app.py
```

It opens a browser tab where you can:

- Upload a `.wav` file
- See predicted emotion and emoji
- Get a matching music recommendation

---

## ✅ Sample .wav Files

You can use open datasets like [RAVDESS](https://zenodo.org/record/1188976) or manually record samples using:

```bash
audacity / online voice recorders / mobile apps
```

Name your files like:

```
happy1.wav, sad1.wav, angry1.wav, etc.
```

---

## 📌 Technologies Used

- Python 3
- Streamlit
- Librosa (audio processing)
- scikit-learn (ML model)
- NumPy / Pandas
- Pickle (for saving models)

---

## 💡 Future Ideas

- Add real-time recording
- Suggest Spotify or JioSaavn playlists
- Improve accuracy using deep learning models
- Deploy using Streamlit Cloud or Hugging Face Spaces

---

## 🙋‍♂️ Author

**Mukul Mehra**  
📫 [mukulmehra681@gmail.com](mailto:mukulmehra681@gmail.com)  
🔗 GitHub: [muks79](https://github.com/muks79)

---

## 📜 License

This project is open-source and free to use under the MIT License.
