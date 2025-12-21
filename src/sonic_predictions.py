# src/lime_explainer.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional
import json

from gradio_client import handle_file
import numpy as np
import pandas as pd
import tensorflow as tf
import soundfile as sf

import librosa
import re

from sonics_api import predict_batch_from_files

def append_update_explanations(new_explanations: dict, explanations_path: Path):
    """
    Adds/update explanations in a JSON file with unified handling of
    full tracks and segments.
    - If entry exists and has non-empty component_influences - do not overwrite
    - If component_influences is empty or entry does not exist - overwrite/add

    :param new_explanations: new explanations dictionary
    :param explanations_path: path to the JSON file
    """

    def is_empty_component_influences(entry: dict) -> bool:
        ci = None
        if entry.get("type") == "full_track":
            ci = entry.get("explanations", {}).get('component_influences')
        elif entry.get("type") == "segment":
            segments = entry.get("segments", {})
            if not segments:
                return True
            for seg_data in segments.values():
                ci_seg = seg_data.get('explanations', {}).get('component_influences')
                if ci_seg is not None and len(ci_seg) > 0:
                    return False
            return True
        else:
            ci = entry.get('component_influences')
        return ci is None or ci == {} or len(ci) == 0

    merged = {}
    if explanations_path.exists():
        try:
            with open(explanations_path, 'r', encoding='utf-8') as f:
                merged = json.load(f)
        except Exception:
            print(f"⚠️ Warning: could not read existing explanations from {explanations_path}")


    for model_name, audio_items in new_explanations.items():
        if model_name not in merged:
            merged[model_name] = audio_items
        else:
            for audio_stem, explanation_data in audio_items.items():
                if audio_stem not in merged[model_name]:
                    merged[model_name][audio_stem] = explanation_data
                else:
                    existing_entry = merged[model_name][audio_stem]

                    if explanation_data.get("type") == "full_track":
                        if is_empty_component_influences(existing_entry):
                            merged[model_name][audio_stem] = explanation_data

                    elif explanation_data.get("type") == "segment":
                        if "segments" not in existing_entry:
                            merged[model_name][audio_stem] = explanation_data
                        else:

                            existing_segments = existing_entry.get("segments", {})
                            new_segments = explanation_data.get("segments", {})

                            for seg_id, seg_expl in new_segments.items():
                                if seg_id not in existing_segments or is_empty_component_influences(existing_segments[seg_id]):
                                    existing_segments[seg_id] = seg_expl

                            merged[model_name][audio_stem]["segments"] = existing_segments

    explanations_path.parent.mkdir(parents=True, exist_ok=True)

    with open(explanations_path, 'w', encoding='utf-8') as f:
        cleaned_data = convert_to_native(merged)
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

def convert_to_native(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def to_native_dict(d):
    native = {}
    for k, v in d.items():
        if isinstance(v, (np.generic, np.ndarray)):
            native[k] = float(v)
        else:
            native[k] = v
    return native
def run_sonics_predictions(
        predictor,
        dataset_path='../../Data/FakeRealMusic',
        explanations_path='predictions.json',
        sample_rate=44100,
        threshold=0.5,
):
    results = {}

    dataset_path = Path(dataset_path)

    for folder in dataset_path.iterdir():
        if not folder.is_dir():
            continue
        class_name = folder.name
        all_audio = list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))
        if not all_audio:
            continue

        print(f"🔊 Processing: {class_name}: {len(all_audio)} files")
        probs = predict_batch_from_files(
            predictor,
            all_audio,
            verbose=True,
            sr=sample_rate
        )

        folder_results = {}
        for audio_file, model_prob in zip(all_audio, probs):
            pred_class = "Fake" if model_prob > threshold else "Real"
            key = audio_file.stem  # file name without extension
            folder_results[key] = {
                "type": "full_track",
                "segment_id": None,
                "predictions": {
                    "file_path": str(audio_file),
                    "model_prediction": float(model_prob),
                    "predicted_class": pred_class
                }
            }
        results[class_name] = folder_results

    output_path = Path(explanations_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Results saved in: {explanations_path}")