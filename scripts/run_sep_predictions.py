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
    dataset_path = "results/AudioLIME/FakeRealMusicOriginal/AudioLIME_500_samples_full_track/full_track/" # ElevenLabs/2__Travis_Scott_Sico_Mode_-_Run_This_Town_Wariant_1_2025-08-22T205303/reversed_separated_components"
    output_json_path = "results/AudioLIME/FakeRealMusicOriginal/AudioLIME_500_samples_full_track/mix_without_components_predictions.json"
    client = Client("awsaf49/sonics-fake-song-detection")

    results = {}

    for model in os.listdir(dataset_path):
        model_folder = os.path.join(dataset_path, model)
        if not os.path.isdir(model_folder):
            continue

        print(f"Processing model: {model}")
        audio_results = {}

        for audio_file in os.listdir(model_folder):
            reversed_separated_components_path = os.path.join(
                model_folder, audio_file, "reversed_separated_components"
            )
            if not os.path.isdir(reversed_separated_components_path):
                continue

            file_results = {}
            audio_stem = Path(audio_file).stem

            for file in os.listdir(reversed_separated_components_path):
                if file.endswith(".wav"):
                    print(f"Processing file: {file}")
                    result = client.predict(
                        audio_file=handle_file(os.path.join(reversed_separated_components_path, file)),
                        model_type="SpecTTTra-α",
                        duration="120s",
                        api_name="/predict",
                    )

                    file_results[file] = {
                        "file_path": os.path.join(reversed_separated_components_path, file),
                        "model": model,
                        "prediction": result,
                    }

            audio_results[audio_stem] = {
                "file_path": str(audio_file),
                "model": model,
                "results": file_results,
            }

        results[model] = audio_results

    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)


        

if __name__ == "__main__":
    main()
