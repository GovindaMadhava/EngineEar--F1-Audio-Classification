import os
import numpy as np
import pickle
import librosa
import warnings
import logging
import tensorflow as tf

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress absl logging warnings
logging.getLogger('absl').setLevel(logging.ERROR)

# Import TensorFlow after setting environment variables
from tensorflow.keras.models import load_model

def extract_features(audio_path, sr=22050, target_shape=(128, 128)):
    """
    Extract mel spectrogram features matching the training process
    """
    try:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return None

        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=30)

        # Check if the audio length is sufficient
        if len(y) < sr * 0.5:
            print(f"Audio file too short: {audio_path}")
            return None

        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or trim to match target shape
        if mel_spec_db.shape[1] < target_shape[1]:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_shape[1] - mel_spec_db.shape[1])), mode='constant')
        elif mel_spec_db.shape[1] > target_shape[1]:
            mel_spec_db = mel_spec_db[:, :target_shape[1]]

        if mel_spec_db.shape != target_shape:
            print(f"Inconsistent mel_spec_db shape: {mel_spec_db.shape}")
            return None

        # Add channel dimension and batch dimension
        mel_spec_db = mel_spec_db[np.newaxis, ..., np.newaxis]
        return mel_spec_db

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def load_models(model_dir='/Users/govindamadhavabs/F1/models'):
    """
    Load all saved models and encoders
    """
    try:
        # Verify model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        print(f"Loading models from: {model_dir}")
        
        # Load the classification models
        team_model_path = os.path.join(model_dir, 'team_classifier.h5')
        driver_model_path = os.path.join(model_dir, 'driver_classifier.h5')
        track_model_path = os.path.join(model_dir, 'track_classifier.h5')
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        
        # Verify all required files exist
        for path in [team_model_path, driver_model_path, track_model_path, encoders_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required model file not found: {path}")
        
        # Custom objects and compile=False to suppress compilation warnings
        team_model = load_model(team_model_path, compile=False)
        driver_model = load_model(driver_model_path, compile=False)
        track_model = load_model(track_model_path, compile=False)
        
        # Compile models with minimal metrics
        for model in [team_model, driver_model, track_model]:
            model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Load the label encoders
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        
        print("Models and encoders loaded successfully")
        return {
            'team_model': team_model,
            'driver_model': driver_model,
            'track_model': track_model,
            'encoders': encoders
        }
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def predict_audio(audio_path, models_dict):
    """
    Make predictions for a given audio file
    """
    try:
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            print("Feature extraction failed.")
            return None
        
        print(f"Input shape for prediction: {features.shape}")
        
        # Get predictions from each model
        team_pred = models_dict['team_model'].predict(features, verbose=0)
        driver_pred = models_dict['driver_model'].predict(features, verbose=0)
        track_pred = models_dict['track_model'].predict(features, verbose=0)
        
        # Get top 3 predictions for each category
        def get_top_3(pred, encoder):
            top_indices = np.argsort(pred[0])[-3:][::-1]
            return [
                {
                    'label': encoder.inverse_transform([idx])[0],
                    'confidence': pred[0][idx] * 100
                }
                for idx in top_indices
            ]
        
        team_top3 = get_top_3(team_pred, models_dict['encoders']['team'])
        driver_top3 = get_top_3(driver_pred, models_dict['encoders']['driver'])
        track_top3 = get_top_3(track_pred, models_dict['encoders']['track'])
        
        return {
            'team': team_top3,
            'driver': driver_top3,
            'track': track_top3
        }
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def format_prediction_result(result):
    """
    Format the prediction results for display
    """
    if result is None:
        return "Error: Could not make prediction"
    
    output = "\nPrediction Results:\n" + "="*50 + "\n"
    
    for category in ['team', 'driver', 'track']:
        output += f"\n{category.title()} Predictions:\n"
        for i, pred in enumerate(result[category], 1):
            output += f"  {i}. {pred['label']} ({pred['confidence']:.2f}%)\n"
    
    return output

def main():
    """
    Main function to run predictions
    """
    try:
        # Load models 
        print("Loading models...")
        models_dict = load_models('/Users/govindamadhavabs/F1/models')
        
        # Test Audio
        '''
        TestAudio2: The test audio is random section taken from the below file 
        Team: Haas Ferrari
        Driver: Kevin Magnussen (#20)
        Track: Alberta Park Australia
        '''
        audio_path = '/Users/govindamadhavabs/F1/test_audio/TestAudio2_HaasFerrari_KevinMagnussen20_AlbertParkCircuitMelbourne^Australia.wav'
        
        
        # Make prediction
        print("\nAnalyzing audio...")
        result = predict_audio(audio_path, models_dict)
        
        # Display results
        print(format_prediction_result(result))
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()