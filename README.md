🎤 Audio Emotion Detection using Machine Learning

This project is a voice-based emotion recognition system built from scratch using Python. It classifies emotions (like happy, sad, angry, etc.) from audio recordings using MFCC features and a Random Forest Classifier, without relying on any pretrained models.
📌 Features

    🔊 Custom Audio Training – Model is trained using your own voice samples.

    🎯 Emotion Classification – Detects basic emotions (e.g., happy, sad, angry).

    🧠 No Pretrained Model Used – Everything is built and trained from scratch.

    📈 Exportable Model – Trained model is saved for future predictions using joblib.

🛠️ Technologies Used

    Python

    scikit-learn – ML model training and evaluation

    librosa – Audio feature extraction (MFCC)

    NumPy & Pandas – Data handling

    joblib – Model serialization

    Jupyter Notebook – Experimentation and visualization

📁 Folder Structure

📦audio-emotion-detector
 ┣ 📂 recordings/           # Raw audio files (.wav)
 ┣ 📂 extracted_features/   # Extracted MFCC features
 ┣ 📜 emotion_model.pkl     # Exported trained model
 ┣ 📜 train.py              # Script to train the model
 ┣ 📜 predict.py            # Predict emotion from a new audio
 ┣ 📜 utils.py              # Helper functions
 ┗ 📜 README.md             # Project documentation

🧪 How It Works

    Record voice samples labeled with emotions (e.g., "happy_1.wav", "sad_1.wav").

    Extract MFCC features from each audio using librosa.

    Train a Random Forest Classifier using scikit-learn.

    Save the trained model using joblib.

    Predict emotion from new voice inputs using the saved model.

▶️ Run the Project
1. Install dependencies

pip install numpy pandas librosa scikit-learn joblib

2. Record or add .wav audio files to /recordings/ folder.
3. Train the model

python train.py

4. Predict emotion from a new voice sample

python predict.py your_audio_file.wav

📊 Sample Emotions (can be customized)

    Happy

    Sad

    Angry

    Neutral

    You can expand the dataset with your own voice samples and labels.

💡 Learning Highlights

    Hands-on experience with audio signal processing

    Understanding MFCC and how voice can be converted into numerical features

    End-to-end supervised ML pipeline

    Model export and prediction without external tools

🧠 Future Improvements

    Add more emotion categories

    Integrate deep learning models (like CNN or LSTM)

    Build a simple web interface for real-time predictions

👨‍💻 Author

Mukul Mehra
Email: mukulmehra681@gmail.com
GitHub: muks79
