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

from audioLIME.factorization_spleeter import SpleeterFactorization
from audioLIME.lime_audio import LimeAudioExplainer
from lime_visualizations import (
    plot_waveforms_overlay_with_influences,
    plot_stacked_rms_area_components,
)

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

def load_existing_explanations(explanations_path: Path):
    if explanations_path.exists():
        try:
            with open(explanations_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            print(f"⚠️ Warning: couldn't load existing explanations from {explanations_path}")
            return {}
    return {}

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

def explain_predictions_segmented(
    audio_path: str,
    predictor,
    segment_duration: float = 10.0,
    model_time: float = 10.0,
    num_samples_lime: int = 500,
    features_output_dir: Optional[str] = None,
    max_samples: int = 5,
    ids_to_explain: Optional[list[int]] = None,
    model_name: Optional[str] = None,
    audio_file_stem: Optional[str] = None,
    max_duration: Optional[float] = None,
    checkpoint_segmented: Optional[LIMEExperimentCheckpoint] = None,
    processed_segments: Optional[Dict[str, List[int]]] = None,
):

    if ids_to_explain is None:
        ids_to_explain = list(range(1000))

    if max_duration is not None:
        y, sr = librosa.load(audio_path, sr=44100, mono=True, duration=max_duration)
    else:
        y, sr = librosa.load(audio_path, sr=44100, mono=True)

    total_duration = len(y) / sr
    total_duration = min(total_duration, model_time)
    segment_samples = int(segment_duration * sr)
    n_segments = int(np.ceil(total_duration / segment_duration))

    print(f'Audio File Stem: {audio_file_stem}')
    print(f'total_duration: {total_duration}')
    print(f'segment_samples: {segment_samples}')
    print(f'n_segments: {n_segments}')

    results = {}

    for seg_i in range(n_segments):

        if processed_segments and seg_i in processed_segments:
            print(f'⏭️ Skipping already processed segment {seg_i}')
            continue

        start_sample = seg_i * segment_samples
        end_sample = min((seg_i + 1) * segment_samples, len(y))
        segment_audio = y[start_sample:end_sample]

        if len(segment_audio) < 2048:
            print(f"⏭️  Segment {seg_i} too short ({len(segment_audio)} samples), skipping.")
            continue

        print(f"\n🎵 Processing segment {seg_i} ({start_sample}-{end_sample} samples, {segment_duration}s approx)")

        segment_dir_name = f"segment_{seg_i}_{int(start_sample / sr)}s_to_{int(end_sample / sr)}s"
        if features_output_dir and model_name and audio_file_stem:
            segment_output_dir = (
                Path(features_output_dir) / model_name / audio_file_stem / segment_dir_name
            )
            segment_output_dir.mkdir(parents=True, exist_ok=True)
            segment_wav_path = segment_output_dir / f"segment_{seg_i}.wav"
            sf.write(segment_wav_path, segment_audio, sr)
        else:
            segment_output_dir = None
            segment_wav_path = None

        fake_prob_segment = predictor.predict(segment_audio, sr=sr)
        predicted_class_segment = 'Fake' if fake_prob_segment > 0.5 else 'Real'

        segment_audio_files = [segment_wav_path] if segment_wav_path else []
        segment_explanations = explain_predictions_separate(
            audio_files=segment_audio_files,
            predictor=predictor,
            model_time=model_time,
            max_samples=max_samples,
            original_predictions=None,
            num_samples_lime=num_samples_lime,
            ids_to_explain=ids_to_explain,
            checkpoint=None,
            folder_name=segment_dir_name,
            explanations_path=None,
            features_output_dir=Path(features_output_dir) / model_name / audio_file_stem
        )

        results[segment_dir_name] = segment_explanations

        if checkpoint_segmented:
            checkpoint_segmented.mark_segment_as_processed(model_name, audio_file_stem, seg_i)

    return results

class LIMEExperimentCheckpoint:
    """
        Manages checkpointing for LIME experiments
        Args:
            checkpoint_dir: Directory to store checkpoint and logs
    """

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "lime_checkpoint.json"
        
    def load_processed_samples(self) -> Dict[str, set]:
        """Load list of already processed samples"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    folder: set(samples) 
                    for folder, samples in data.get('processed_samples', {}).items()
                }
        return {}
    
    def mark_as_processed(self, folder: str, sample_id: int):
        """Mark sample as processed"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'processed_samples': {}}
        
        if folder not in data['processed_samples']:
            data['processed_samples'][folder] = []
        
        if sample_id not in data['processed_samples'][folder]:
            data['processed_samples'][folder].append(sample_id)
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            cleaned_data = convert_to_native(data)
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    def load_processed_segments(self) -> Dict[str, Dict[str, List[int]]]:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("processed_segments", {})
        return {}

    def mark_segment_as_processed(self, model: str, audio_file_stem: str, segment_idx: int):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'processed_segments': {}}
        if model not in data['processed_segments']:
            data['processed_segments'][model] = {}
        if audio_file_stem not in data['processed_segments'][model]:
            data['processed_segments'][model][audio_file_stem] = []
        if segment_idx not in data['processed_segments'][model][audio_file_stem]:
            data['processed_segments'][model][audio_file_stem].append(segment_idx)
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def predict_fn_unified(waveforms, predictor):
    """
    Unified prediction function for LIME (works with Local and Remote).

    :param waveforms: Array of waveforms [N, samples] or [samples]
    :param predictor: LocalSonnics or RemoteSonnics instance
    :return: Array of probabilities [N, 2] (real_prob, fake_prob)
    """
    
    if waveforms.ndim == 1:
        waveforms = waveforms[np.newaxis, :]

    probs = []
    for waveform in waveforms:
        fake_prob = predictor.predict(waveform, sr=44100)
        real_prob = 1 - fake_prob
        probs.append([real_prob, fake_prob])
    
    return np.array(probs)

def explain_predictions_separate(
    audio_files, 
    predictor,
    model_time, 
    max_samples=5, 
    original_predictions=None,
    num_samples_lime=500,
    ids_to_explain=[0,1,2,3,4,5,6,7,8,9],
    checkpoint: Optional[LIMEExperimentCheckpoint] = None,
    folder_name: str = "",
    explanations_path: Optional[str] = None,
    features_output_dir: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Function to explain predictions for a set of audio files using LIME.
    """

    print(f"Starting explanation for {min(len(audio_files), max_samples)} samples...")

    audio_arrays = []
    for fpath in audio_files[:max_samples]:
        y, _ = librosa.load(fpath, sr=44100, mono=True, offset=0, duration=model_time)
        audio_arrays.append(y)

    processed_samples = set()
    if checkpoint:
        all_processed = checkpoint.load_processed_samples()
        processed_samples = all_processed.get(folder_name, set())
    
    sample_info = {}

    if features_output_dir:
        features_path = Path(features_output_dir) / folder_name / "features.json"
    
        if features_path.exists():
            try:
                with open(features_path, 'r', encoding='utf-8') as f:
                    features_all_tracks = json.load(f)
            except Exception:
                print(f"⚠️ Warning: Could not load existing physical features from {features_path}")

    if explanations_path:
        explanations_path_obj  = Path(explanations_path)
        if explanations_path_obj.exists():
            try:
                with open(explanations_path_obj, 'r', encoding='utf-8') as f:
                    sample_info = json.load(f)
            except Exception:
                print(f"⚠️ Warning: Cannot load existing explanations from {explanations_path}")

    for i, waveform in enumerate(audio_arrays):
        new_explanations = None
        if i >= max_samples:
            break

        if i not in ids_to_explain: 
            continue

        if i in processed_samples:
            print(f"⏭️  Sample {i+1} already processed, skipping...")
            continue
            
        print(f"\n🔍 Processing sample {i+1}/{len(audio_arrays)}...")

        influences = {}
        error_message = ""

        try:
            if original_predictions is not None:
                fake_prob = original_predictions[i]
                is_fake = fake_prob > 0.5
                print(f"🔮 Model prediction: {fake_prob:.4f} ({'Fake' if is_fake else 'Real'})")
            else:
                fake_prob = None
                is_fake = None

            with tf.Graph().as_default():
                if waveform.ndim > 1:
                    waveform_mono = waveform[0]
                else:
                    waveform_mono = waveform
                
                factorization = SpleeterFactorization(
                    input=waveform_mono,
                    target_sr=44100,
                    temporal_segmentation_params=1,
                    composition_fn=None,
                    model_name="spleeter:4stems"
                )

                explainer = LimeAudioExplainer(kernel_width=0.25)
                
                explanation = explainer.explain_instance(
                    factorization=factorization,
                    predict_fn=lambda x: predict_fn_unified(x, predictor),
                    num_samples=num_samples_lime,
                    top_labels=1
                )
                
                weights = explanation.local_exp[explanation.top_labels[0]]
                influences = {
                    component: weight[1] 
                    for component, weight in zip(factorization._components_names, weights)
                }

                top_components = sorted(influences.items(), key=lambda x: abs(x[1]), reverse=True)
                
                if fake_prob is None:
                    predicted_class = None
                else:
                    predicted_class = 'Fake' if fake_prob > 0.5 else 'Real'

                sample_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(audio_files[i]).stem)

                sample_info[sample_key] = {
                    'file_path': str(audio_files[i]),
                    'model_prediction': fake_prob,
                    'predicted_class': predicted_class,
                    'component_influences': influences,
                }

                orig_filename = Path(audio_files[i]).stem
                safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', orig_filename)

                if features_output_dir:

                    features_audio_output_dir = Path(features_output_dir or "") / folder_name / safe_name
                    features_audio_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    plot_waveforms_overlay_with_influences(
                        original_audio=waveform_mono,
                        components=factorization.components,
                        component_names=factorization._components_names,
                        influences=influences,
                        sr=44100,
                        output_path=features_audio_output_dir,
                        prefix=safe_name
                    )
                    
                    plot_stacked_rms_area_components(
                        components=factorization.components,
                        component_names=factorization._components_names,
                        influences=influences,
                        sr=44100,
                        output_path=features_audio_output_dir,
                        prefix=safe_name
                    )

                print(f"✅ Finished processing sample {i+1}.")

                print(f"📊 Components influence:")
                for component, weight in influences.items():
                    print(f"  {component}: {weight:.4f}")

                new_explanations = {
                    model_name: {
                        safe_name: {
                        "track_id": i,
                        "type": "full_track",
                        "segment_id": None,
                        "explanations": sample_info[sample_key]
                        }   
                    }
                }

                if explanations_path and model_name:
                    append_update_explanations(new_explanations, Path(explanations_path))
                    print(f'💾 Saved explanation for sample {i+1} to {explanations_path}')

                if checkpoint:
                    checkpoint.mark_as_processed(folder_name, i)
                    
        except Exception as e:
            if new_explanations and explanations_path:
                append_update_explanations(new_explanations, Path(explanations_path))
            error_message = str(e)
            print(f"❌ Error processing sample {i+1}: {error_message}")

    return sample_info

def run_lime_experiment_safe(
    predictor,
    model_time=120, 
    explain=False, 
    max_samples_explain=5, 
    dataset_path='../../Data/FakeRealMusic', 
    num_samples_lime=500,
    models_to_explain=['ElevenLabs', 'REAL', 'SUNO', 'SUNO_PRO', 'UDIO'],
    ids_to_explain=[0,1,2,3,4,5,6,7,8,9],
    checkpoint_dir: Optional[str | Path] = None,
    explanations_path: Optional[str] = None,
    features_output_dir_full: Optional[str] = None,
    features_output_dir_segmented: Optional[str] = None,
    full_track_explanations: bool = True,
    segmented_explanations: bool = False,
    segment_duration: float = 10.0,
    segmented_explanations_path: Optional[str] = None,
):
    """ 
    Run the LIME experiment for fake song detection.
    """
    
    checkpoint = None
    if checkpoint_dir and explain:
        checkpoint = LIMEExperimentCheckpoint(checkpoint_dir)
    if checkpoint_dir and segmented_explanations and explain:
        checkpoint_segmented = LIMEExperimentCheckpoint(str(checkpoint_dir) + "_segmented")
    results = {}

    merged_explanations = {}
    if explanations_path:
        merged_explanations = load_existing_explanations(Path(explanations_path))
        print(f"Loaded existing explanations for {len(merged_explanations)} folders from {explanations_path}")

    if segmented_explanations and segmented_explanations_path:
        merged_segmented_explanations = load_existing_explanations(Path(segmented_explanations_path))
    else:
        merged_segmented_explanations = {}

    for folder in Path(dataset_path).iterdir():
        if not folder.is_dir() or folder.name not in models_to_explain:
            continue
            
        print(f"\n🔊 Processing folder: {folder.name}")
        all_audio = list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))
        all_audio = sorted(all_audio)[:max_samples_explain]
        
        if not all_audio:
            print(f"   No audio files found, skipping...")
            continue
        
        print(f"   Getting predictions for {len(all_audio)} files...")
        
        original_probs = predict_batch_from_files(
            predictor, 
            all_audio, 
            verbose=True,
            sr=44100,  # For local
            duration=model_time  # For local
        )
        
        predictions = [prob > 0.5 for prob in original_probs]
        results[folder.name] = predictions
        
        if explain:
            if full_track_explanations:
                folder_explanations = explain_predictions_separate(
                    audio_files=all_audio,
                    predictor=predictor,
                    model_time=model_time,
                    max_samples=max_samples_explain,
                    original_predictions=original_probs,
                    num_samples_lime=num_samples_lime,
                    ids_to_explain=ids_to_explain,
                    checkpoint=checkpoint,
                    folder_name=folder.name,
                    explanations_path=explanations_path,
                    features_output_dir=features_output_dir_full,
                    model_name=folder.name
                )

                folder_explanations_str_keys = {
                    str(k): {
                        "type": "full_track",
                        "segment_id": None,
                        "explanations": v
                    }
                    for k, v in folder_explanations.items()
                    if not isinstance(k, int) and not (isinstance(k, str) and k.isdigit())
                }

                if not folder_explanations_str_keys:
                    print(f"   No explanations found for folder: {folder.name}")
                
            if segmented_explanations:
                print(f"\n🔊 Processing segmented explanations for folder: {folder.name}")

                processed_segments = checkpoint_segmented.load_processed_segments() if checkpoint_segmented else {}

                if not merged_segmented_explanations.get(folder.name):
                    merged_segmented_explanations[folder.name] = {}

                for audio_file in all_audio:
                    audio_stem = Path(audio_file).stem

                    segment_explanations = explain_predictions_segmented(
                        audio_path=str(audio_file),
                        predictor=predictor,
                        segment_duration=segment_duration,
                        model_time=model_time,
                        num_samples_lime=num_samples_lime,
                        features_output_dir=features_output_dir_segmented,
                        max_samples=max_samples_explain,
                        model_name = folder.name,
                        audio_file_stem=audio_stem,
                        max_duration=model_time,
                        checkpoint_segmented=checkpoint_segmented,
                        processed_segments=processed_segments.get(folder.name, {}).get(audio_stem, []),
                    )

                    if folder.name not in merged_segmented_explanations:
                        merged_segmented_explanations[folder.name] = {}
                    if audio_stem not in merged_segmented_explanations[folder.name]:
                        merged_segmented_explanations[folder.name][audio_stem] = {
                            "type": "segment",
                            "segments": {}
                        }
                
                    filtered_segment_explanations = {
                        str(k): v for k, v in segment_explanations.items() if not (isinstance(k, str) and k.isdigit())
                    }

                    for segment_id, explanations_data in filtered_segment_explanations.items():
                        merged_segmented_explanations[folder.name][audio_stem]["segments"][segment_id] = {
                            "explanations": explanations_data
                        }
                    if segmented_explanations_path:
                        append_update_explanations(merged_explanations, Path(segmented_explanations_path))

    df = pd.DataFrame(results)
    print("\n📊 Results DataFrame (True = Fake):")
    print(df)
    
    if explain:
        if explanations_path and Path(explanations_path).exists():
            return df, load_existing_explanations(Path(explanations_path))
        else:
            return df, {}
    else:
        return df