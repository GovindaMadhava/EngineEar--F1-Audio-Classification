# Formula1-EngineAudio-Classifier
The first comprehensive audio dataset containing high-quality F1 engine audio for the complete 2024 Formula 1 season.

Dataset Stats:

25 Tracks | 20 Drivers | 10 Teams | 25,640 Laps | 423 GBs / 615 hours of stereo .wav audio data sampled at 22.050 Hz

Link to Kaggle dataset:-

Formula 1 Pole dap dataset: https://www.kaggle.com/datasets/govindamadhava/formula1-f1-pure-engine-audio-dataset-2012-2024 

Formula 1 Grand Prix Engine Audio: https://www.kaggle.com/datasets/govindamadhava/formula-1-grand-prix-engine-audio

Model:

Developed a Deep Learning pipeline for multi-class classification of Engine exhaust sounds to its corresponding Team/Engine, Track, Driver. 
Leveraging a large-scale curated dataset of race audio segmented by lap, the approach extracts Mel-spectrograms, MFCCs and Chroma features to train a CNN-BiLSTM architecture. A novel team-driver awareness function is employed to improve the driver prediction accuracy.

Our best models achieve the following figures:

Team model:- Accuracy: 99.24% | Loss: 0.0321 | F1 Score: 0.9924

Driver Model:- Accuracy: 87.76% | Loss: 0.4541 | F1 Score: 0.8769

Track Model:- Accuracy: 91.81% | Loss: 0.2925 | F1 Score: 0.9179

How to use the models and predict on an audio of your choice??

1. Download the best team, track and driver models and label encoder file found in this repository.
2. Update the model paths in the formula1_prediction.ipynb Jupyter notebook.
3. Use any .wav audio file or one from the kaggle dataset and update the path to the audio file.
4. The output will be the top 3 preedictions for the 3 categories + the pitch contour of the input audio file.


Share your comments, reviews and suggestions: gmbs.madhava001@gmail.com
