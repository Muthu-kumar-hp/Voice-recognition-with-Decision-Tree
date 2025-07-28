## 🎤 Voice Recognition Prediction Web App

This is a voice-enabled web application built with **Flask**, **SpeechRecognition**, and **scikit-learn**. The app takes voice input from the user (e.g., "Male 23"), processes the audio, and predicts an outcome using a trained **Decision Tree Classifier**.

---

## 🧠 Features

- 🎙️ Capture input using microphone
- 🔊 Use Google Speech Recognition API to extract text
- 🔍 Predict using a trained Decision Tree model
- 🌐 Simple and responsive web interface
- 📦 Reusable LabelEncoders for preprocessing
- 🧪 Microphone testing script

---

## 📁 Project Structure

voice_prediction_app/
│
├── app.py # Main Flask app
├── train.py # Script to train and save the model and encoders
├── mic_test.py # Test script for voice input
├── model.pkl # Trained Decision Tree model
├── encoder_age.pkl # LabelEncoder for age
├── encoder_gender.pkl # LabelEncoder for gender
│
├── templates/
│ ├── index.html # Web form for voice input
│ └── result.html # Displays prediction result
│
├── requirements.txt # List of required Python packages
└── README.md # Project documentation

---

## 📦 Requirements

Install the dependencies using pip:

```bash
pip install -r requirements.txt
Flask
SpeechRecognition
pyaudio
scikit-learn
pandas
joblib
🖼️ HTML Templates
index.html:
Contains a button to trigger voice capture.

result.html:
Displays the predicted output returned by the model.
🧠 Machine Learning
The train.py script uses DecisionTreeClassifier to train the model on a sample dataset (you can replace this with your own). It uses LabelEncoder for handling categorical inputs like gender and age groups.
✅ Future Improvements
 Add offline voice recognition support using vosk

 Improve voice command validation

 Add additional features like education level or location

📄 License
This project is licensed under the MIT License.
Free to use, modify, and distribute.
