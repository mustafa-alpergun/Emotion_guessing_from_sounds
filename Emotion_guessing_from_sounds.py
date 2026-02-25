import os
import logging
import joblib
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioEmotionClassifier:
    """Class for emotion analysis from audio files."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self) -> MLPClassifier:
        """Creates the MLP model and sets hyperparameters."""
        return MLPClassifier(
            alpha=0.01,
            batch_size=256,
            epsilon=1e-08,
            hidden_layer_sizes=(300,),
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )

    def extract_features(self, file_path: str) -> np.ndarray:
        """Extracts MFCC, Chroma, and Mel features from the specified audio file."""
        try:
            X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            
            # Feature extraction processes
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            
            return np.hstack((mfccs, chroma, mel))
        except Exception as e:
            logging.error(f"Error occurred while reading file ({file_path}): {e}")
            return None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reads the dataset from the directory and returns X, y arrays."""
        X, y = [], []
        logging.info("Reading data and extracting features. This may take a while...")
        
        if not self.data_path.exists():
            logging.error(f"Specified data path not found: {self.data_path}")
            return np.array(X), np.array(y)

        for folder in self.data_path.iterdir():
            if folder.is_dir():
                for file_path in folder.glob('*.wav'):
                    # Parsing emotion code from the filename (e.g., 03-01-05-01... -> 05)
                    emotion_code = file_path.name.split('-')[2]
                    features = self.extract_features(str(file_path))
                    
                    if features is not None:
                        X.append(features)
                        y.append(emotion_code)
                        
        return np.array(X), np.array(y)

    def train_and_evaluate(self):
        """Trains the model and evaluates it on test data."""
        X, y = self.load_data()
        
        if len(X) == 0:
            logging.error("No data found for training. Terminating process.")
            return

        logging.info(f"Total of {len(X)} audio files processed successfully.")
        logging.info("Splitting dataset into training (%80) and testing (%20)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scaling data (Standardization) is critical for the MLP algorithm
        logging.info("Scaling data with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logging.info("Training the MLP model...")
        self.model.fit(X_train_scaled, y_train)

        logging.info("Predicting on test data...")
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model Training Completed. Test Accuracy: {accuracy * 100:.2f}%\n")
        print("Classification Report:\n")
        print(classification_report(y_test, y_pred))

    def save_model(self, model_filename: str = "emotion_model.pkl", scaler_filename: str = "scaler.pkl"):
        """Saves the trained model and scaler object to disk."""
        try:
            joblib.dump(self.model, model_filename)
            joblib.dump(self.scaler, scaler_filename)
            logging.info(f"Model and scaler saved successfully: {model_filename}, {scaler_filename}")
        except Exception as e:
            logging.error(f"Error occurred while saving the model: {e}")

if __name__ == "__main__":
    # Main execution block
    DATA_DIRECTORY = r"C:\Users\muham\Downloads\archive (5)\audio_speech_actors_01-24"
    
    classifier = AudioEmotionClassifier(data_path=DATA_DIRECTORY)
    classifier.train_and_evaluate()
    classifier.save_model()