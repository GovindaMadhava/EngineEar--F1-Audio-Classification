'''
23Nov24:
GUI to record audio from pc mic + display results
'''
# pip install sounddevice soundfile tkinter
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import librosa
import sounddevice as sd
import soundfile as sf
import datetime
import time
from threading import Thread
import queue
import tkinter as tk
from tkinter import ttk
import wave

class F1AudioPredictor:
    def __init__(self, model_dir='/Users/govindamadhavabs/F1/old_models'):
        # Audio recording parameters
        self.SAMPLE_RATE = 44100
        self.CHANNELS = 1
        self.RECORDING_LENGTH = 5  # seconds
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Load models and encoders
        self.MODEL_DIR = model_dir
        self.load_models()
        
    def load_models(self):
        """Load all necessary models and encoders"""
        self.team_model = load_model(os.path.join(self.MODEL_DIR, 'old_team_classifier.h5'))
        self.driver_model = load_model(os.path.join(self.MODEL_DIR, 'Formula1_driver_classifier.h5'))
        self.track_model = load_model(os.path.join(self.MODEL_DIR, 'old_track_classifier.h5'))
        self.encoder = joblib.load(os.path.join(self.MODEL_DIR, 'model_data.pkl'))
        
    def extract_features(self, audio_file):
        """Extract features from audio file"""
        y, sr = librosa.load(audio_file, sr=None, mono=True)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, max(0, 128 - log_mel_spec.shape[1]))), mode='constant')
        log_mel_spec = log_mel_spec[:, :128]
        return np.expand_dims(log_mel_spec, axis=-1)
    
    def predict(self, model, features, label_encoder):
        """Make prediction using the specified model"""
        probabilities = model.predict(np.expand_dims(features, axis=0))[0]
        prediction = np.argmax(probabilities)
        confidence = np.max(probabilities)
        return label_encoder.inverse_transform([prediction])[0], confidence
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if status:
            print(f"Error in audio recording: {status}")
        self.audio_queue.put(indata.copy())
    
    def save_recording(self, filename):
        """Save recorded audio to WAV file"""
        data = []
        while not self.audio_queue.empty():
            data.append(self.audio_queue.get())
        
        if not data:
            return None
            
        audio_data = np.concatenate(data)
        sf.write(filename, audio_data, self.SAMPLE_RATE)
        return filename
    
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        self.audio_queue = queue.Queue()
        
        try:
            with sd.InputStream(callback=self.audio_callback,
                              channels=self.CHANNELS,
                              samplerate=self.SAMPLE_RATE):
                print("Recording started...")
                time.sleep(self.RECORDING_LENGTH)
                
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
            
        self.is_recording = False
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"live_recording_{timestamp}.wav"
        return self.save_recording(filename)
    
    def predict_audio(self, audio_file):
        """Make predictions for audio file"""
        features = self.extract_features(audio_file)
        
        team, team_conf = self.predict(self.team_model, features, self.encoder['encoders']['team'])
        driver, driver_conf = self.predict(self.driver_model, features, self.encoder['encoders']['driver'])
        track, track_conf = self.predict(self.track_model, features, self.encoder['encoders']['track'])
        
        return {
            'team': (team, team_conf),
            'driver': (driver, driver_conf),
            'track': (track, track_conf)
        }

class F1PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("F1 Audio Predictor")
        self.predictor = F1AudioPredictor()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Recording button
        self.record_button = ttk.Button(main_frame, text="Start Recording", command=self.record_and_predict)
        self.record_button.grid(row=0, column=0, pady=10)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=1, column=0, pady=5)
        
        # Results display
        self.results_text = tk.Text(main_frame, height=10, width=50)
        self.results_text.grid(row=2, column=0, pady=10)
        
    def record_and_predict(self):
        """Handle recording and prediction process"""
        self.record_button.state(['disabled'])
        self.status_label.config(text="Recording...")
        self.results_text.delete(1.0, tk.END)
        
        # Start recording in a separate thread
        def record_thread():
            audio_file = self.predictor.start_recording()
            if audio_file:
                self.status_label.config(text="Analyzing...")
                results = self.predictor.predict_audio(audio_file)
                
                # Display results
                result_text = "Prediction Results:\n\n"
                for category, (prediction, confidence) in results.items():
                    result_text += f"{category.title()}: {prediction} ({confidence*100:.2f}%)\n"
                
                self.results_text.insert(tk.END, result_text)
                self.status_label.config(text="Ready")
            else:
                self.status_label.config(text="Recording failed!")
            
            self.record_button.state(['!disabled'])
        
        Thread(target=record_thread, daemon=True).start()

def main():
    root = tk.Tk()
    app = F1PredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()