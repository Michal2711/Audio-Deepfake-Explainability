#   dataset_path: "results/AudioLIME/FakeMusicOriginal/AudioLIME_500_samples_full_track/REAL/SaintLevant_IGuess_mp3/reversed_separated_components"
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('h5py').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

import argparse
from pathlib import Path
import yaml
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonics_api import LocalSonnics, RemoteSonnics
from gradio_client import Client, handle_file

def main():
    orig_data = "Data/FakeRealMusicOriginal/ElevenLabs/2._Travis_Scott_Sico_Mode_-_Run_This_Town_Wariant_1_2025-08-22T205303.mp3"
    dataset_path = "results/AudioLIME/FakeRealMusicOriginal/AudioLIME_500_samples_full_track/full_track/ElevenLabs/2__Travis_Scott_Sico_Mode_-_Run_This_Town_Wariant_1_2025-08-22T205303/reversed_separated_components"

    client = Client("awsaf49/sonics-fake-song-detection")

    result = client.predict(
                audio_file=handle_file(orig_data),
                model_type = "SpecTTTra-α",
                duration="120s",
                api_name="/predict"
    )
    print(f"Prediction result for {orig_data}: {result}")

    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            print(f"Processing file: {file}")
            result = client.predict(
                audio_file=handle_file(os.path.join(dataset_path, file)),
                model_type = "SpecTTTra-α",
                duration="120s",
                api_name="/predict")
            print(f"Prediction result for {file}: {result}")

if __name__ == "__main__":
    main()

# orig: 0.929344892501831
# bass0.wav: 0.4380253553390503
# drums0.wav: 0.08595184236764908
# other0.wav:  0.12505245208740234
# vocals0.wav: 0.27443310618400574
            
# Prediction result for Data/FakeRealMusicOriginal/REAL/SaintLevant_IGuess.mp3.mp3: {'label': 'Fake', 'confidences': [{'label': 'Fake', 'confidence': 0.929344892501831}, {'label': 'Real', 'confidence': 0.07065510749816895}]}
# Processing file: bass0.wav
# Prediction result for bass0.wav: {'label': 'Fake', 'confidences': [{'label': 'Fake', 'confidence': 0.5619746446609497}, {'label': 'Real', 'confidence': 0.4380253553390503}]}
# Processing file: drums0.wav
# Prediction result for drums0.wav: {'label': 'Real', 'confidences': [{'label': 'Real', 'confidence': 0.9140481352806091}, {'label': 'Fake', 'confidence': 0.08595184236764908}]}
# Processing file: other0.wav
# Prediction result for other0.wav: {'label': 'Fake', 'confidences': [{'label': 'Fake', 'confidence': 0.8749475479125977}, {'label': 'Real', 'confidence': 0.12505245208740234}]}
# Processing file: vocals0.wav
# Prediction result for vocals0.wav: {'label': 'Real', 'confidences': [{'label': 'Real', 'confidence': 0.7255668640136719}, {'label': 'Fake', 'confidence': 0.27443310618400574}]}

pred: 0.78
bass0: 0.91
drums0: 0.40
other0: 0.47
vocals0: 0.89