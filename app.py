from flask import Flask, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model_path = 'saved_models/speech_emotion_recognition.hdf5'  # Adjust the path accordingly
model = load_model(model_path)

# Add any necessary preprocessing functions
def preprocess_audio(audio_path):
    sample_rate = 22050  # Adjust according to your model's requirements
    audio, _ = librosa.load(audio_path, sr=sample_rate, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    mfccs_scaled_features = mfccs_scaled_features[:, :, np.newaxis]
    return mfccs_scaled_features

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the file temporarily or process it directly
        # Example: file.save('temp_audio.wav')
        # processed_data = preprocess_audio('temp_audio.wav')

        # Use the loaded model to make predictions
        predictions = model.predict(processed_data)
        predicted_label = predictions.argmax(axis=1)[0]

        # Convert the label to a human-readable emotion (adjust as needed)
        emotions = ['emotion1', 'emotion2', 'emotion3']  # Replace with your actual emotion labels
        predicted_emotion = emotions[predicted_label]

        return jsonify({'predicted_emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
