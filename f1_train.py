'''
Team Classification:
- Number of teams: 11 (ASTONMARTIN twice)
- Test Accuracy: 40.62%
- Test Loss: 3.0768

Driver Classification:
- Number of drivers: 23 (LANDO, NORRIS, OSCAR, PIASTRI -> BEARMAN extra)
- Test Accuracy: 37.50%
- Test Loss: 2.9240

Track Classification:
- Number of tracks: 5
- Test Accuracy: 53.12%
- Test Loss: 3.0884
epoch 8/8, 100

FINAL MODEL SUMMARY
==================================================
Epoch 100/100
140/140 ━━━━━━━━━━━━━━━━━━━━ 98s 696ms/step - accuracy: 0.9883 - loss: 0.0337 - val_accuracy: 0.9284 - val_loss: 0.3189
Team Classification:
- Number of teams: 10
- Test Accuracy: 92.84%
- Test Loss: 0.3189

Epoch 100/100
140/140 ━━━━━━━━━━━━━━━━━━━━ 89s 632ms/step - accuracy: 0.9665 - loss: 0.1241 - val_accuracy: 0.5551 - val_loss: 2.8805
Driver Classification:
- Number of drivers: 22
- Test Accuracy: 55.51%
- Test Loss: 2.8805

Epoch 100/100
140/140 ━━━━━━━━━━━━━━━━━━━━ 89s 633ms/step - accuracy: 0.9569 - loss: 0.1272 - val_accuracy: 0.8048 - val_loss: 0.9055
Track Classification:
- Number of tracks: 6
- Test Accuracy: 80.48%
- Test Loss: 0.9055
'''
import os
import numpy as np
import pandas as pd
import pickle
import librosa
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM, Reshape, TimeDistributed
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import csv

# Filter out warnings
warnings.filterwarnings('ignore', category=UserWarning)

def extract_features(audio_path, sr=22050, n_mfcc=13, target_shape=(128, 128)):
    """
    Extract comprehensive audio features for F1 car audio classification.
    
    Features extracted:
    1. Mel Spectrogram: Visual representation of frequency content over time
    2. MFCCs (Mel-frequency cepstral coefficients): Represent the timbre and vocal tract shape
    3. Spectral Contrast: Difference between peaks and valleys in the spectrum
    4. Chroma Features: Representation of pitch content
    5. Zero Crossing Rate: Rate of signal changing from positive to negative
    6. Spectral Centroid: Center of mass of the spectrum (brightness)
    7. Spectral Bandwidth: Width of the range of frequencies
    8. RMS Energy: Volume/energy of the signal
    9. Spectral Rolloff: Frequency below which most of the energy lies
    
    Args:
        audio_path: Path to the audio file
        sr: Sampling rate (default: 22050 Hz)
        n_mfcc: Number of MFCC coefficients
        target_shape: Target shape for mel spectrogram
        
    Returns:
        Dictionary containing all extracted features
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

        # Pad or trim the mel spectrogram to the target shape
        if mel_spec_db.shape[1] < target_shape[1]:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_shape[1] - mel_spec_db.shape[1])), mode='constant')
        elif mel_spec_db.shape[1] > target_shape[1]:
            mel_spec_db = mel_spec_db[:, :target_shape[1]]

        if mel_spec_db.shape != target_shape:
            print(f"Inconsistent mel_spec_db shape after padding for {audio_path}: {mel_spec_db.shape}")
            return None

        return mel_spec_db

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def prepare_data_from_csv(csv_path):
    """
    Prepare data from CSV file with the new column structure.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None, None, None

    mel_specs = []
    teams = []
    drivers = []
    tracks = []

    try:
        # Read CSV with pandas for better handling
        df = pd.read_csv(csv_path)
        print(f"CSV columns found: {df.columns.tolist()}")
        
        for idx, row in df.iterrows():
            print(f"\nProcessing row {idx + 1}:")
            
            # Get values from correct columns
            audio_path = row['File Path']
            team = row['Team Name']
            driver = row['Driver Name']
            track = row['Track Name']
            
            if not isinstance(audio_path, str) or not audio_path.strip():
                print(f"Row {idx + 1}: Empty or invalid audio path")
                continue

            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                continue

            # Extract features
            mel_spec_db = extract_features(audio_path)
            if mel_spec_db is not None:
                mel_specs.append(mel_spec_db)
                teams.append(team)
                drivers.append(driver)
                tracks.append(track)

        print("\nProcessing Summary:")
        print(f"Total rows processed: {len(df)}")
        print(f"Successful extractions: {len(mel_specs)}")

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    if not mel_specs:
        print("\nNo features extracted!")
        return None, None, None, None

    # Reshape mel spectrograms for CNN input
    mel_specs = np.array(mel_specs)[..., np.newaxis]  # Add channel dimension
    return mel_specs, teams, drivers, tracks
def build_classifier(input_shape, num_classes):
    """
    Build CNN-RNN model for classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Reshape((-1, 128)),  # Reshape for RNN layers

        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models(mel_specs, teams, drivers, tracks):
    """
    Train separate models for team, driver, and track classification with detailed summaries.
    """
    # Label encoders
    team_encoder = LabelEncoder()
    driver_encoder = LabelEncoder()
    track_encoder = LabelEncoder()

    # Encode labels
    team_encoded = to_categorical(team_encoder.fit_transform(teams))
    driver_encoded = to_categorical(driver_encoder.fit_transform(drivers))
    track_encoded = to_categorical(track_encoder.fit_transform(tracks))

    # Split data
    test_size = 0.2
    random_state = 42

    X_train, X_test, y_team_train, y_team_test = train_test_split(mel_specs, team_encoded, test_size=test_size, random_state=random_state)
    _, _, y_driver_train, y_driver_test = train_test_split(mel_specs, driver_encoded, test_size=test_size, random_state=random_state)
    _, _, y_track_train, y_track_test = train_test_split(mel_specs, track_encoded, test_size=test_size, random_state=random_state)

    # Print dataset split information
    print("\nDataset Split Information:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    models = {}

    # Train team model
    print("\n" + "="*50)
    print("Training Team Classification Model")
    print("="*50)
    team_model = build_classifier((128, 128, 1), len(team_encoder.classes_))
    print("\nTeam Model Architecture:")
    team_model.summary()
    team_history = team_model.fit(
        X_train, y_team_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_team_test),
        verbose=1
    )
    
    # Evaluate team model
    team_loss, team_acc = team_model.evaluate(X_test, y_team_test, verbose=0)
    print(f"\nTeam Model Final Test Accuracy: {team_acc*100:.2f}%")
    print(f"Team Model Final Test Loss: {team_loss:.4f}")
    print(f"Number of teams: {len(team_encoder.classes_)}")
    print("Teams:", ', '.join(team_encoder.classes_))

    # Train driver model
    print("\n" + "="*50)
    print("Training Driver Classification Model")
    print("="*50)
    driver_model = build_classifier((128, 128, 1), len(driver_encoder.classes_))
    print("\nDriver Model Architecture:")
    driver_model.summary()
    driver_history = driver_model.fit(
        X_train, y_driver_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_driver_test),
        verbose=1
    )
    
    # Evaluate driver model
    driver_loss, driver_acc = driver_model.evaluate(X_test, y_driver_test, verbose=0)
    print(f"\nDriver Model Final Test Accuracy: {driver_acc*100:.2f}%")
    print(f"Driver Model Final Test Loss: {driver_loss:.4f}")
    print(f"Number of drivers: {len(driver_encoder.classes_)}")
    print("Drivers:", ', '.join(driver_encoder.classes_))

    # Train track model
    print("\n" + "="*50)
    print("Training Track Classification Model")
    print("="*50)
    track_model = build_classifier((128, 128, 1), len(track_encoder.classes_))
    print("\nTrack Model Architecture:")
    track_model.summary()
    track_history = track_model.fit(
        X_train, y_track_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_track_test),
        verbose=1
    )
    
    # Evaluate track model
    track_loss, track_acc = track_model.evaluate(X_test, y_track_test, verbose=0)
    print(f"\nTrack Model Final Test Accuracy: {track_acc*100:.2f}%")
    print(f"Track Model Final Test Loss: {track_loss:.4f}")
    print(f"Number of tracks: {len(track_encoder.classes_)}")
    print("Tracks:", ', '.join(track_encoder.classes_))

    # Print final summary of all models
    print("\n" + "="*50)
    print("FINAL MODEL SUMMARY")
    print("="*50)
    print("\nTeam Classification:")
    print(f"- Number of teams: {len(team_encoder.classes_)}")
    print(f"- Test Accuracy: {team_acc*100:.2f}%")
    print(f"- Test Loss: {team_loss:.4f}")
    
    print("\nDriver Classification:")
    print(f"- Number of drivers: {len(driver_encoder.classes_)}")
    print(f"- Test Accuracy: {driver_acc*100:.2f}%")
    print(f"- Test Loss: {driver_loss:.4f}")
    
    print("\nTrack Classification:")
    print(f"- Number of tracks: {len(track_encoder.classes_)}")
    print(f"- Test Accuracy: {track_acc*100:.2f}%")
    print(f"- Test Loss: {track_loss:.4f}")

    return {
        'team_model': team_model,
        'driver_model': driver_model,
        'track_model': track_model,
        'encoders': {'team': team_encoder, 'driver': driver_encoder, 'track': track_encoder},
        'histories': {
            'team': team_history.history,
            'driver': driver_history.history,
            'track': track_history.history
        }
    }

def save_models(models_dict, save_dir='models'):
    """
    Save models, encoders, and training histories.
    """
    if models_dict is None:
        print("No models to save!")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    # Save models
    models_dict['team_model'].save(os.path.join(save_dir, 'team_classifier.h5'))
    models_dict['driver_model'].save(os.path.join(save_dir, 'driver_classifier.h5'))
    models_dict['track_model'].save(os.path.join(save_dir, 'track_classifier.h5'))

    # Save encoders
    with open(os.path.join(save_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(models_dict['encoders'], f)

    # Save training histories
    with open(os.path.join(save_dir, 'training_histories.pkl'), 'wb') as f:
        pickle.dump(models_dict['histories'], f)

    print(f"\nAll models, encoders, and training histories saved to {save_dir}")

if __name__ == "__main__":
    csv_path = '/Users/govindamadhavabs/Desktop/f1_lap_dataset.csv'
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        exit(1)

    print("\nStarting data preparation...")
    print(f"Using CSV file: {csv_path}")
    
    # First, let's peek at the CSV file
    try:
        with open(csv_path, 'r') as f:
            print("\nFirst few lines of CSV file:")
            for i, line in enumerate(f):
                if i < 5:  # Print first 5 lines
                    print(f"Line {i}: {line.strip()}")
                else:
                    break
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        exit(1)

    # Prepare data with error handling
    result = prepare_data_from_csv(csv_path)
    
    if result[0] is None:
        print("Error: Failed to prepare data. Exiting.")
        exit(1)
        
    mel_specs, teams, drivers, tracks = result

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(mel_specs)}")
    print(f"Unique teams: {len(set(teams))}")
    print(f"Unique drivers: {len(set(drivers))}")
    print(f"Unique tracks: {len(set(tracks))}")

    # Train models
    try:
        models_dict = train_models(mel_specs, teams, drivers, tracks)
        save_models(models_dict)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        exit(1)