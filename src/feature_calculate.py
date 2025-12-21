# src/feature_calculate.py
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

from feature_extraction import (
    extract_all_features,
)

from feature_visualizations import (
    plot_all_waveforms, 
    plot_all_spectrograms,
    plot_rms_envelope,
    plot_rhythm_statistics,
    plot_enhanced_visualizations,
    plot_f0_contour,
    plot_mel_spectrogram_with_f0,
    plot_spectral_summary,
)

def append_update_features(new_features: dict, features_path: Path):
    """
    Merged new features with existing ones in a JSON file.
    Full track and segmented handling.
    """

    def is_empty_features(entry: dict) -> bool:
        if entry.get("type") == "full_track":
            return not bool(entry.get("features"))
        elif entry.get("type") == "segment":
            segments = entry.get("segments", {})
            if not segments:
                return True
            for seg_data in segments.values():
                if seg_data.get("features"):
                    return False
            return True
        else:
            return not bool(entry.get("features"))

    merged = {}
    if features_path.exists():
        try:
            with open(features_path, 'r', encoding='utf-8') as f:
                merged = json.load(f)
        except Exception:
            print(f"⚠️ Warning: could not read existing features from {features_path}")

    for model_name, audio_items in new_features.items():
        if model_name not in merged:
            merged[model_name] = audio_items
        else:
            for audio_stem, feature_data in audio_items.items():
                if audio_stem not in merged[model_name]:
                    merged[model_name][audio_stem] = feature_data
                else:
                    existing_entry = merged[model_name][audio_stem]

                    if feature_data.get("type") == "full_track":
                        if is_empty_features(existing_entry):
                            merged[model_name][audio_stem] = feature_data

                    elif feature_data.get("type") == "segment":
                        if "segments" not in existing_entry:
                            merged[model_name][audio_stem] = feature_data
                        else:
                            existing_segments = existing_entry.get("segments", {})
                            new_segments = feature_data.get("segments", {})

                            for seg_id, seg_features in new_segments.items():
                                if seg_id not in existing_segments or is_empty_features(existing_segments[seg_id]):
                                    existing_segments[seg_id] = seg_features

                            merged[model_name][audio_stem]["segments"] = existing_segments

    features_path.parent.mkdir(parents=True, exist_ok=True)
    with open(features_path, 'w', encoding='utf-8') as f:
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

def extract_all_features_separately(
    audio_files: List[Path],
    max_samples: int = 5,
    ids_to_get_features: Optional[List[int]] = [0,1,2,3,4,5,6,7,8,9],
    features_output_dir: Optional[Path] = None,
    folder_name: str = "",
    sample_rate: int = 44100
):
    
    print(f"Starting calculating features for {min(len(audio_files), max_samples)} samples...")
    all_features = {}
    audio_arrays = []
    for fpath in audio_files[:max_samples]:
        y, _ = librosa.load(fpath, sr=sample_rate, mono=True, offset=0, duration=120)
        audio_arrays.append(y)

    features_all_tracks = {}

    if features_output_dir:
        features_path = Path(features_output_dir) / folder_name / "features.json"

        if features_path.exists():
            try:
                with open(features_path, 'r', encoding='utf-8') as f:
                    features_all_tracks = json.load(f)
            except Exception as e:
                print(f"   Warning: Could not load existing features from {features_path}: {e}")
    
    for i, waveform in enumerate(audio_arrays):
        if i >= max_samples:
            break
        if i not in ids_to_get_features:
            continue

        features_for_track = {}

        print(f"   Extracting features from sample {i+1}/{len(audio_arrays)}")

        try:
            with tf.Graph().as_default():
                if waveform.ndim > 1:
                    waveform_mono = librosa.to_mono(waveform)
                else:
                    waveform_mono = waveform

                factorization = SpleeterFactorization(
                    input=waveform_mono,
                    target_sr=sample_rate,
                    temporal_segmentation_params=1,
                    composition_fn=None,
                    model_name="spleeter:4stems"
                )

                orig_filename = Path(audio_files[i]).stem
                safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', orig_filename)
                features_audio_output_dir = Path(features_output_dir or "") / folder_name / safe_name
                features_audio_output_dir.mkdir(parents=True, exist_ok=True)

                mix_features = extract_all_features(waveform_mono, sr=sample_rate)

                if 'intonation_pattern' in mix_features:
                        mix_features['intonation_pattern'].pop('f0_contour', None)
                        mix_features['intonation_pattern'].pop('times', None)

                if 'rhythm_stats' in mix_features:
                    mix_features['rhythm_stats'].pop('beats_times', None)

                features_for_track['mix'] = to_native_dict(mix_features)

                if features_output_dir:
                    plot_rms_envelope(waveform_mono, sr=sample_rate,
                    output_path=features_audio_output_dir / f"{safe_name}_rms_envelope.png",
                    title=f"RMS Envelope ({safe_name})")

                    rhythm_stats = plot_rhythm_statistics(
                        waveform_mono, sr=sample_rate,
                        output_dir=features_audio_output_dir, prefix=safe_name
                    )

                    plot_spectral_summary(
                        waveform_mono,
                        sr=sample_rate,
                        output_dir=features_audio_output_dir,
                        prefix=safe_name
                    )

                    features_audio_output_dir = Path(features_output_dir or "") / folder_name / safe_name
                    features_audio_output_dir.mkdir(parents=True, exist_ok=True)

                    plot_all_waveforms(
                        original_audio=waveform_mono,
                        components=factorization.components,
                        component_names=factorization._components_names,
                        sr=sample_rate,
                        output_path=features_audio_output_dir,
                        prefix=safe_name
                    )
                    
                    plot_all_spectrograms(
                        original_audio=waveform_mono,
                        components=factorization.components,
                        component_names=factorization._components_names,
                        sr=sample_rate,
                        output_path=features_audio_output_dir,
                        prefix=safe_name
                    )
                    
                for idx, comp_name in enumerate(factorization._components_names):
                    component_audio = factorization.components[idx]

                    component_audio_dir = Path(features_audio_output_dir or "") / comp_name
                    component_audio_dir.mkdir(parents=True, exist_ok=True)
                    
                    all_features = extract_all_features(component_audio, sr=sample_rate)

                    features_for_track[comp_name] = to_native_dict(all_features)

                    plot_enhanced_visualizations(
                        component_audio,
                        sr=sample_rate,
                        prefix=f"{safe_name}_{comp_name}",
                        output_dir=component_audio_dir
                    )

                    plot_spectral_summary(
                        component_audio,
                        sr=sample_rate,
                        output_dir=component_audio_dir,
                        prefix=f"{safe_name}_{comp_name}"
                    )

                    f0_contour = np.array(all_features.get('intonation_pattern', {}).get('f0_contour', []))
                    pitch_times = np.array(all_features.get('intonation_pattern', {}).get('times', []))
                    f0_contour_clean = np.nan_to_num(f0_contour, nan=0.0)

                    if 'intonation_pattern' in all_features:
                        all_features['intonation_pattern'].pop('f0_contour', None)
                        all_features['intonation_pattern'].pop('times', None)

                    if 'rhythm_stats' in all_features:
                        all_features['rhythm_stats'].pop('beats_times', None)

                    plot_f0_contour(
                        y=component_audio,
                        sr=sample_rate,
                        f0=f0_contour_clean,
                        times=pitch_times,
                        title=f"{safe_name}_{comp_name} - Fundamental Frequency Contour",
                        output_dir=component_audio_dir,
                        prefix=f"{safe_name}_{comp_name}"
                    )

                    plot_mel_spectrogram_with_f0(
                        y=component_audio,
                        sr=sample_rate,
                        f0=f0_contour_clean,
                        times=pitch_times,
                        title=f"{safe_name}_{comp_name} - Mel Spectrogram with Fundamental Frequency Contour",
                        output_dir=component_audio_dir,
                        prefix=f"{safe_name}_{comp_name}"
                    )
                features_all_tracks[safe_name] = features_for_track

        except Exception as e:
            error_message = str(e)
            print(f"❌ Error processing sample {i+1}: {error_message}")

    return features_all_tracks

def extract_features_segmented(
    audio_path: str,
    max_duration: Optional[float] = None,
    ids_to_get_features: Optional[List[int]] = None,
    model_time: float = 120.0,
    features_output_dir: Optional[Path] = None,
    max_samples: int = 5,
    model_name: Optional[str] = None,
    audio_file_stem: Optional[str] = None,
    segment_duration: float = 10.0,
    sample_rate: int = 44100,
) -> Dict[str, Dict]:
    if ids_to_get_features is None:
        ids_to_get_features = list(range(1000))
    
    if max_duration is not None:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_duration)
    else:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    total_duration = len(y) / sr
    total_duration = min(total_duration, model_time)
    segment_samples = int(segment_duration * sr)
    n_segments = int(np.ceil(total_duration / segment_duration))

    print(f'total_duration: {total_duration}')
    print(f'segment_samples: {segment_samples}')
    print(f'n_segments: {n_segments}')

    results = {}

    for seg_i in range(n_segments):
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
                   
        segmented_audio_files = [segment_wav_path] if segment_wav_path else []
        segmented_features = extract_all_features_separately(
            audio_files=segmented_audio_files,
            max_samples=max_samples,
            ids_to_get_features=ids_to_get_features,
            features_output_dir=Path(features_output_dir) / model_name / audio_file_stem,
            folder_name=segment_dir_name,
            sample_rate=sample_rate,
        )

        results[segment_dir_name] = segmented_features

    return results

def run_features_extraction(
    dataset_path: Path,
    model_time = 120.0,
    max_samples=5, 
    models_to_get_features=['ElevenLabs', 'REAL', 'SUNO', 'SUNO_PRO', 'UDIO'],
    ids_to_get_features=[0,1,2,3,4,5,6,7,8,9],
    features_output_dir_full: Optional[str] = None,
    features_output_dir_segmented: Optional[str] = None,
    full_track_features=True,
    segmented_features=False,
    segment_duration=5.0,
    sample_rate=44100,
    ):

    merged_segmented_features = {}

    for folder in Path(dataset_path).iterdir():
        if not folder.is_dir() or folder.name not in models_to_get_features:
            continue

        print(f"\n🔊 Processing folder: {folder.name}")
        all_audio = list(folder.glob("*.mp3")) + list(folder.glob("*.wav"))
        all_audio = sorted(all_audio)[:max_samples]
        
        if not all_audio:
            print(f"   No audio files found, skipping...")
            continue

        if full_track_features:
            folder_features = extract_all_features_separately(
                audio_files=all_audio,
                max_samples=max_samples,
                ids_to_get_features=ids_to_get_features,
                features_output_dir=features_output_dir_full,
                folder_name=folder.name
            )

            folder_features_str_keys = {
                str(k): {
                    "type": "full_track",
                    "segment_id": None,
                    "features": v
                }
                for k, v in folder_features.items()
                if not isinstance(k, int) and not (isinstance(k, str) and k.isdigit())
            }
            
            if not folder_features_str_keys:
                continue

            append_update_features({folder.name: folder_features_str_keys}, Path(features_output_dir_full / "features_full_track.json"))

        if segmented_features:
            merged_segmented_features[folder.name] = {}

            for audio_file in all_audio:
                audio_stem = Path(audio_file).stem
            
                segmented_features = extract_features_segmented(
                    audio_path=str(audio_file),
                    segment_duration=segment_duration,
                    features_output_dir=features_output_dir_segmented,
                    max_samples=max_samples,
                    ids_to_get_features=ids_to_get_features,
                    model_name=folder.name,
                    audio_file_stem=audio_stem,
                    max_duration=model_time,
                    sample_rate=sample_rate
                )

                if folder.name not in merged_segmented_features:
                    merged_segmented_features[folder.name] = {}
                if audio_stem not in merged_segmented_features[folder.name]:
                    merged_segmented_features[folder.name][audio_stem] = {
                        "type": "segment",
                        "segments": {}
                    }

                filtered_segment_features = {
                    str(k): v for k, v in segmented_features.items() if not (isinstance(k, str) and k.isdigit())
                }

                for segment_id, features_data in filtered_segment_features.items():
                    merged_segmented_features[folder.name][audio_stem]["segments"][segment_id] = {
                        "features": features_data
                    }

                append_update_features(merged_segmented_features, Path(features_output_dir_segmented / "segmented_features.json"))

