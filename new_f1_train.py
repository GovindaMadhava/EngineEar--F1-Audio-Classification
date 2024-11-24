'''
none are performing good

Team:
Epoch 42: val_accuracy did not improve from 0.96958
321/321 ━━━━━━━━━━━━━━━━━━━━ 236s 734ms/step - accuracy: 0.9817 - loss: 0.0622 - val_accuracy: 0.9637 - val_loss: 0.1691 - learning_rate: 1.0000e-05

Driver:
Epoch 15: val_accuracy did not improve from 0.80460
321/321 ━━━━━━━━━━━━━━━━━━━━ 203s 633ms/step - accuracy: 0.7803 - loss: 1.0166 - val_accuracy: 0.7391 - val_loss: 1.1546 - learning_rate: 0.0010

Track:
Epoch 7: val_accuracy did not improve from 0.20983
321/321 ━━━━━━━━━━━━━━━━━━━━ 503s 2s/step - accuracy: 0.2071 - loss: 2.4346 - val_accuracy: 0.1841 - val_loss: 2.4674 - learning_rate: 0.0010

Epoch 15: val_accuracy did not improve from 0.25702
321/321 ━━━━━━━━━━━━━━━━━━━━ 453s 1s/step - accuracy: 0.2684 - loss: 2.1352 - val_accuracy: 0.2250 - val_loss: 2.2534 - learning_rate: 2.0000e-04
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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tqdm import tqdm
import traceback
import tensorflow as tf

# Filter out warnings
warnings.filterwarnings('ignore', category=UserWarning)

def extract_features(audio_path, sr=22050, target_shape=(128, 128)):
    """Extract mel spectrogram features from audio"""
    try:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return None

        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=30)

        if len(y) < sr * 0.5:
            print(f"Audio file too short: {audio_path}")
            return None

        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or trim to target shape
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
    """Prepare data from CSV file"""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None, None, None, None, None

    mel_specs = []
    teams = []
    drivers = []
    tracks = []
    
    # Create team-driver mapping
    team_driver_map = {}
    
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV columns found: {df.columns.tolist()}")
        
        # Build team-driver mapping first
        for _, row in df.iterrows():
            team = row['Team Name']
            driver = row['Driver Name']
            if team not in team_driver_map:
                team_driver_map[team] = set()
            team_driver_map[team].add(driver)
        
        for idx, row in tqdm(df.iterrows(), desc="Processing audio files"):
            audio_path = row['File Path']
            team = row['Team Name']
            driver = row['Driver Name']
            track = row['Track Name']
            
            if not isinstance(audio_path, str) or not audio_path.strip():
                continue

            if not os.path.exists(audio_path):
                continue

            mel_spec_db = extract_features(audio_path)
            if mel_spec_db is not None:
                mel_specs.append(mel_spec_db)
                teams.append(team)
                drivers.append(driver)
                tracks.append(track)

        # Convert team-driver map to dictionary of lists
        team_driver_map = {team: list(drivers) for team, drivers in team_driver_map.items()}

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

    if not mel_specs:
        print("No features extracted!")
        return None, None, None, None, None

    mel_specs = np.array(mel_specs)[..., np.newaxis]
    return mel_specs, teams, drivers, tracks, team_driver_map

def build_team_classifier(input_shape, num_classes):
    """Build team classifier (keeping the successful architecture from old_train.py)"""
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
        
        Reshape((-1, 128)),
        
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def build_track_classifier(input_shape, num_classes):
    """Build improved track classifier with enhanced architecture"""
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Reshape((-1, 256)),
        
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        LSTM(64, kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def build_driver_classifier(input_shape, num_classes):
    """Build driver classifier using architecture from f1_train.py"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.40),
        
        Conv2D(64, (3, 3), activation='relu', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.40),
        
        Conv2D(128, (3, 3), activation='relu', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.50),
        
        Reshape((-1, 128)),
        
        LSTM(64, return_sequences=True, 
             kernel_regularizer=l2(0.001)),
        Dropout(0.40),
        LSTM(32, kernel_regularizer=l2(0.001)),
        Dropout(0.40),
        
        Dense(128, activation='relu', 
              kernel_regularizer=l2(0.001)),
        Dropout(0.50),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

class TeamAwareDriverPredictor:
    """Custom predictor that considers team context for driver predictions"""
    def __init__(self, team_model, driver_model, team_encoder, driver_encoder, team_driver_map):
        self.team_model = team_model
        self.driver_model = driver_model
        self.team_encoder = team_encoder
        self.driver_encoder = driver_encoder
        self.team_driver_map = team_driver_map
        
        self.team_idx_to_name = {idx: name for idx, name in enumerate(team_encoder.classes_)}
        self.driver_idx_to_name = {idx: name for idx, name in enumerate(driver_encoder.classes_)}
        self.driver_name_to_idx = {name: idx for idx, name in enumerate(driver_encoder.classes_)}
        
    def predict(self, X):
        team_pred = self.team_model.predict(X)
        driver_pred_raw = self.driver_model.predict(X)
        adjusted_driver_pred = np.zeros_like(driver_pred_raw)
        
        for i in range(len(X)):
            predicted_team_idx = np.argmax(team_pred[i])
            predicted_team = self.team_idx_to_name[predicted_team_idx]
            valid_drivers = self.team_driver_map[predicted_team]
            valid_driver_indices = [self.driver_name_to_idx[driver] for driver in valid_drivers]
            
            adjusted_driver_pred[i] = np.zeros(len(self.driver_encoder.classes_))
            valid_probs = driver_pred_raw[i][valid_driver_indices]
            valid_probs = valid_probs / np.sum(valid_probs)
            
            for idx, prob in zip(valid_driver_indices, valid_probs):
                adjusted_driver_pred[i][idx] = prob
                
        return adjusted_driver_pred

def train_models(mel_specs, teams, drivers, tracks, team_driver_map):
    """Train and evaluate all models"""
    team_encoder = LabelEncoder()
    driver_encoder = LabelEncoder()
    track_encoder = LabelEncoder()
    
    team_encoded = to_categorical(team_encoder.fit_transform(teams))
    driver_encoded = to_categorical(driver_encoder.fit_transform(drivers))
    track_encoded = to_categorical(track_encoder.fit_transform(tracks))
    
    # Split data
    test_size = 0.2
    random_state = 42
    
    X_train, X_test, y_team_train, y_team_test = train_test_split(
        mel_specs, team_encoded, test_size=test_size, random_state=random_state)
    _, _, y_driver_train, y_driver_test = train_test_split(
        mel_specs, driver_encoded, test_size=test_size, random_state=random_state)
    _, _, y_track_train, y_track_test = train_test_split(
        mel_specs, track_encoded, test_size=test_size, random_state=random_state)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )
    
    # Train team model
    print("\nTraining Team Classification Model")
    team_model = build_team_classifier((128, 128, 1), len(team_encoder.classes_))
    team_checkpoint = ModelCheckpoint(
        'team_model_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    team_history = team_model.fit(
        X_train, y_team_train,
        validation_data=(X_test, y_team_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, team_checkpoint],
        verbose=1
    )
    
    # Train driver model
    print("\nTraining Driver Classification Model")
    driver_model = build_driver_classifier((128, 128, 1), len(driver_encoder.classes_))
    driver_checkpoint = ModelCheckpoint(
        'driver_model_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    driver_history = driver_model.fit(
        X_train, y_driver_train,
        validation_data=(X_test, y_driver_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, driver_checkpoint],
        verbose=1
    )
    
    # Create team-aware predictor
    team_aware_predictor = TeamAwareDriverPredictor(
        team_model, driver_model, team_encoder, driver_encoder, team_driver_map
    )
    
    # Train track model
    print("\nTraining Track Classification Model")
    track_model = build_track_classifier((128, 128, 1), len(track_encoder.classes_))
    track_checkpoint = ModelCheckpoint(
        'track_model_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    track_history = track_model.fit(
        X_train, y_track_train,
        validation_data=(X_test, y_track_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, track_checkpoint],
        verbose=1
    )
    
    return {
        'team_model': team_model,
        'driver_model': driver_model,
        'track_model': track_model,
        'team_aware_predictor': team_aware_predictor,
        'team_driver_map': team_driver_map,
        'encoders': {'team': team_encoder, 'driver': driver_encoder, 'track': track_encoder},
        'histories': {
            'team': team_history.history,
            'driver': driver_history.history,
            'track': track_history.history
        }
    }

def save_models(models_dict, save_dir='/Users/govindamadhavabs/F1/models'):
    """Save all models and related data"""
    if models_dict is None:
        print("No models to save!")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    # Save models with .keras extension
    try:
        models_dict['team_model'].save(os.path.join(save_dir, 'New_Formula1_team_classifier.keras'))
        models_dict['driver_model'].save(os.path.join(save_dir, 'New_Formula1_driver_classifier.keras'))
        models_dict['track_model'].save(os.path.join(save_dir, 'New_Formula1_track_classifier.keras'))
    except Exception as e:
        print(f"Error saving models: {e}")
        return
    
    # Save encoders and additional data
    try:
        with open(os.path.join(save_dir, 'model_data.pkl'), 'wb') as f:
            data_to_save = {
                'encoders': models_dict['encoders'],
                'team_driver_map': models_dict['team_driver_map'],
                'histories': models_dict['histories']
            }
            pickle.dump(data_to_save, f)
    except Exception as e:
        print(f"Error saving additional data: {e}")
        return

    print(f"\nAll models and data saved to {save_dir}")



if __name__ == "__main__":
    csv_path = '/Users/govindamadhavabs/Desktop/F1_lap_dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        exit(1)

    print("\nStarting data preparation...")
    result = prepare_data_from_csv(csv_path)
    
    if result[0] is None:
        print("Error: Failed to prepare data. Exiting.")
        exit(1)
        
    mel_specs, teams, drivers, tracks, team_driver_map = result
    
    try:
        models_dict = train_models(mel_specs, teams, drivers, tracks, team_driver_map)
        save_models(models_dict)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        exit(1)