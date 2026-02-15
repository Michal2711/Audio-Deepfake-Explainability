# src/dsp_band_ops.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from datetime import datetime
import time
from typing import Iterable, List, NamedTuple, Tuple, Optional, Dict, Any

import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns

try:
    from audioLIME.factorization_spleeter import SpleeterFactorization
except Exception:
    SpleeterFactorization = None

class TimeAggregator:
    def __init__(self):
        self.global_stats = {}
        self.sample_stats = {}

    def record(self, name: str, elapsed: float):
        self.global_stats.setdefault(name, []).append(elapsed)
        self.sample_stats.setdefault(name, []).append(elapsed)

    def reset_sample(self):
        self.sample_stats = {}

    def summary(self, stats: dict[str, list[float]]):
        out = {}
        for name, values in stats.items():
            total = sum(values)
            count = len(values)
            avg = total / count if count > 0 else 0.0
            out[name] = {
                "total": total,
                "count": count,
                "avg": avg
            }
        return out

    def print_sample_summary(self):
        if not self.sample_stats:
            return
        print("\n⏱️ Sample timing summary:")
        for name, s in self.summary(self.sample_stats).items():
            print(f"  - {name}: total {s['total']:.2f}s, calls {s['count']}, avg {s['avg']:.3f}s")

    def print_global_summary(self):
        if not self.global_stats:
            return
        print("\n⏱️ Global timing summary:")
        for name, s in self.summary(self.global_stats).items():
            print(f"  - {name}: total {s['total']:.2f}s, calls {s['count']}, avg {s['avg']:.3f}s")

def timed(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            profiler = getattr(self, "profiler", None)
            start = time.time()
            try:
                return func(self, *args, **kwargs)
            finally:
                elapsed = time.time() - start
                if profiler is not None:
                    profiler.record(name, elapsed)
        return wrapper
    return decorator


def append_update_fbp_results(
    new_results: dict, 
    results_path: Path
) -> None:
    """
    Structure:
    {
        "ModelA": {
            "file1": { ...result... },
            "file2": { ... }
        },
        "ModelB": { ... }
    }
    """
    merged: dict = {}

    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                merged = json.load(f)
        except Exception:
            print(f"⚠️ Warning: could not read existing spectrogram results from {results_path}")
            merged = {}

    for model_name, files_dict in new_results.items():
        if model_name not in merged:
            merged[model_name] = {}
        for file_key, data in files_dict.items():
            merged[model_name][file_key] = data

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

class ExperimentCheckpoint:
    """
        Manages checkpointing and logging of experiment progress
        Args:
            checkpoint_dir: Directory to store checkpoint and logs
    """
    
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "processing_checkpoint.json"
        self.failed_files_log = self.checkpoint_dir / "failed_files.json"
        self.progress_log = self.checkpoint_dir / "progress.txt"
        
    def load_processed_files(self) -> set:
        """Load the set of already processed files"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('processed_files', []))
        return set()
    
    def mark_as_processed(self, file_path: str, success: bool = True, error_msg: str = None):
        """Mark a file as processed"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'processed_files': [], 'last_updated': None, 'total_processed': 0, 'total_failed': 0}

        if file_path not in data['processed_files']:
            data['processed_files'].append(file_path)
            data['total_processed'] = len(data['processed_files'])
        
        data['last_updated'] = datetime.now().isoformat()
        
        if not success:
            data['total_failed'] = data.get('total_failed', 0) + 1

        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if not success and error_msg:
            self._log_failed_file(file_path, error_msg)
    
    def _log_failed_file(self, file_path: str, error_msg: str):
        """Log information about failed processing"""
        if self.failed_files_log.exists():
            with open(self.failed_files_log, 'r', encoding='utf-8') as f:
                failed = json.load(f)
        else:
            failed = {'failed_files': []}
        
        failed['failed_files'].append({
            'file_path': file_path,
            'error': str(error_msg),
            'timestamp': datetime.now().isoformat()
        })
        
        with open(self.failed_files_log, 'w', encoding='utf-8') as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

        with open(self.progress_log, 'a', encoding='utf-8') as f:
            f.write(f"[FAILED] {datetime.now().isoformat()} | {file_path} | {error_msg}\n")
    
    def get_failed_files(self) -> list:
        """Return list of files that failed"""
        if self.failed_files_log.exists():
            with open(self.failed_files_log, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('failed_files', [])
        return []
    
    def get_stats(self) -> dict:
        """Return processing statistics"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {
                    'total_processed': data.get('total_processed', 0),
                    'total_failed': data.get('total_failed', 0),
                    'last_updated': data.get('last_updated', None)
                }
        return {'total_processed': 0, 'total_failed': 0, 'last_updated': None}
    
    def reset(self):
        """Reset checkpoint - remove all saved data"""
        for f in [self.checkpoint_file, self.failed_files_log, self.progress_log]:
            if f.exists():
                f.unlink()

# Preset frequency bands (in Hz)
FREQUENCY_BAND_PRESETS: Dict[str, List[Tuple[int, int]]] = {
    "default": [
        (20, 100), (100, 250), (250, 2000),
        (2000, 4000), (4000, 8000), (8000, 16000)
    ],
    "detailed_voice": [
        (20, 60), (60, 250), (250, 500), (500, 2000),
        (2000, 4000), (4000, 6000), (6000, 12000), (12000, 21000)
    ],
    "high_resolution": [
        (20, 60), (60, 100), (100, 250), (250, 500), (500, 1000), (1000, 2000),
        (2000, 4000), (4000, 6000), (6000, 8000), (8000, 10000), (10000, 12000),
        (12000, 16000), (16000, 21000)
    ],
}

def match_rms(ref: np.ndarray, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    r_ref = float(np.sqrt(np.mean(ref**2) + eps))
    r_x = float(np.sqrt(np.mean(x**2) + eps))
    if r_x < eps:
        return x
    return x * (r_ref / r_x)


def smooth_band_keep_mask(freqs: np.ndarray, low: float, high: float, trans: float = 200.0) -> np.ndarray:
    """
    Returns a keep mask (1 = keep, 0 = mute) with smooth edges.
    The band [low, high] is faded to 0; outside the band the mask is ~1.
    trans: width of the transition area on both sides in Hz.
    """
    f = freqs.astype(float)
    m = np.ones_like(f, dtype=float)

    core = (f >= low) & (f <= high)
    m[core] = 0.0

    if trans > 0:
        tl = (f >= (low - trans)) & (f < low)
        if np.any(tl):
            x = (f[tl] - (low - trans)) / trans
            m[tl] = 0.5 * (1.0 + np.cos(np.pi * x))  # 1 -> 0

        th = (f > high) & (f <= (high + trans))
        if np.any(th):
            x = (f[th] - high) / trans
            m[th] = 0.5 * (1.0 + np.cos(np.pi * (1.0 - x)))  # 0 -> 1

    return np.clip(m, 0.0, 1.0)

def tf_retry_decorator(max_retries: int = 20):
    import functools, time, traceback

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    msg = str(e).lower()
                    indicators = ('tensor', 'graph', 'out of scope', 'tensorflow', 'tensordataset')
                    if any(ind in msg for ind in indicators):
                        print(f"[Warning] TF graph scope error on attempt {attempt+1} in {func.__name__}")

                        try:
                            import tensorflow as tf
                            import gc
                            tf.keras.backend.clear_session()
                            gc.collect()
                        except Exception as clear_err:
                            print(f"[Warning] Could not clear TF session: {clear_err}")
                        
                        if attempt < max_retries - 1:
                            time.sleep(1.0)
                            continue
                    raise
                    
            print(f"[Error] Failed to complete {func.__name__} after {max_retries} attempts.")
            return pd.DataFrame() if func.__name__ == "process_audio_file" else None
        return wrapper
    return decorator

# class Predictor:
#     """
#     Minimal interface required by FrequencyBandPerturbation.
#     Implementations: LocalSonnics / RemoteSonnics in src/model_api.py.
#     """
#     def predict(self, audio_wave: np.ndarray, sr: int) -> float:
#         raise NotImplementedError

class FBDResult(NamedTuple):
    importance_map: Optional[np.ndarray]
    spectrogram_db: np.ndarray
    baseline_pred: float
    y: np.ndarray
    S: np.ndarray
    batch_importances: Optional[list[dict]]

class FrequencyBandPerturbation:
    def __init__(self, 
                 predictor: Predictor,
                 preset: str = "default",
                 presets: Optional[Dict[str, List[Tuple[int, int]]]] = None,
                 attenuation: float = 0.0,
                 transition_mode: str = "rel",
                 transition_hz: float = 0.0,
                 transition_rel: float = 0.0,
                 transition_min_hz: float = 0.0, 
                 transition_max_hz: float = 0.0,
                 sr: int = 44100,
                 duration: int = 120,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: int = 2048,
                 n_iter: int = 256,
                 spec_type: str = "stft",
                 fmax: Optional[float] = None,
                 use_original_audio: bool = False,
                 use_separation: bool = False,
                 separation_model: str = "spleeter:2stems",
                 separation_targets: Tuple[str, ...] = ("vocals0", "accompaniment0"),
                 normalize_loudness: bool = True,
                 lufs: Optional[float] = None,
                 checkpoint_dir: Optional[str | Path] = None
    ):
        self.predictor = predictor
        self.preset = preset
        self.presets = presets

        if self.presets is not None:
            self.bands = self.presets.get(self.preset, FREQUENCY_BAND_PRESETS["default"])
        else:
            self.bands = FREQUENCY_BAND_PRESETS.get(self.preset, FREQUENCY_BAND_PRESETS["default"])

        self.attenuation = attenuation
        self.transition_mode = transition_mode
        self.transition_hz = transition_hz
        self.transition_rel = transition_rel
        self.transition_min_hz = transition_min_hz
        self.transition_max_hz = transition_max_hz

        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_iter = n_iter

        self.spec_type = spec_type.lower()
        if self.spec_type not in ("stft"):
            raise ValueError("FrequencyBandPerturbation currently supports only spec_type='stft'")

        self.fmax = fmax if fmax is not None else sr // 2

        self.use_original_audio = use_original_audio
        self.use_separation = use_separation
        self.separation_model = separation_model
        self.separation_targets = separation_targets

        self.normalize_loudness = normalize_loudness
        self.lufs = lufs

        self.profiler = TimeAggregator()

        if checkpoint_dir:
            self.checkpoint = ExperimentCheckpoint(checkpoint_dir)

    @timed("Computing spectrogram")
    def _compute_spectrogram(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  
        """Return (S, S_db) for current spec_type"""
        if self.spec_type == "mel":
            S = librosa.feature.melspectrogram(
                y=y, 
                sr=self.sr, 
                n_mels=self.n_mels, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                win_length=self.win_length, 
                fmax=self.fmax
            )
            S_db = librosa.power_to_db(S, ref=np.max)
        else:
            S = librosa.stft(
                y, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length,
                window='hann',
                center=True
            )
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        return S, S_db
    
    @timed("Inverting spectrogram")
    def _invert_spectrogram(self, S: np.ndarray) -> np.ndarray:
        """Invert spectrogram to audio using current spec_type"""
        if self.spec_type == "mel":
            y_rec = librosa.feature.inverse.mel_to_audio(
                S, 
                sr=self.sr, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                win_length=self.win_length, 
                n_iter=self.n_iter
            )
        else:
            y_rec = librosa.istft(
                S, 
                hop_length=self.hop_length, 
                win_length=self.win_length,
                window='hann',
                center=True
            )
        return y_rec

    def _band_transition_width(self, low: float, high: float) -> float:
        bw = float(high - low)
        if self.transition_mode == "rel":
            trans = bw * self.transition_rel
            trans = float(np.clip(trans, self.transition_min_hz, self.transition_max_hz))
        else:
            trans = float(self.transition_hz)
        return trans

    @timed("Predicting audio")
    def _predict(self, wave: np.ndarray) -> float:
        "Wrapper for predictor"
        try:
            with torch.no_grad():
                return float(self.predictor.predict(wave, self.sr))
        except Exception as e:
            print(f"[Warning] Prediction error: {type(e).__name__}: {e}")
            return 0.0

    @timed("Separating sources")
    def _separate_sources(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.use_separation:
            return {"mixture": audio}

        if SpleeterFactorization is None:
            print("[Warning] Spleeter not available, falling back to mixture.")
            return {"mixture": audio}

        fact = SpleeterFactorization(
            input=audio,
            target_sr=self.sr,
            temporal_segmentation_params=1,
            composition_fn=None,
            model_name=self.separation_model,
        )
        return dict(zip(fact._components_names, fact.original_components))

    def _save_frequency_band_importances(
        self,
        y: np.ndarray,
        S: np.ndarray,
        batch_importances: list[dict],
        file_name: str,
        save_dir: Path,
        save_audio: bool = True
    ):
        save_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "file_name": file_name,
            "bands": []
        }

        # Przygotuj częstotliwości odpowiadające indeksom w S
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        for idx, p in enumerate(batch_importances, 1):
            low = p["low"]
            high = p["high"]
            importance = p["importance"]
            component = p.get("component", "mixture")
            component_dir = save_dir / "freq_batches"
            component_dir.mkdir(parents=True, exist_ok=True)

            if self.use_original_audio:
                y_band = y.copy()
            else:
                mag, phase = librosa.magphase(S)
                band_mask = (freqs >= low) & (freqs <= high)
                mag_band = np.zeros_like(mag)
                mag_band[band_mask, :] = mag[band_mask, :]

                S_band = mag_band * phase
                y_band = self._invert_spectrogram(S_band)
                
            importance_type = "POSITIVE" if importance > 0 else "NEGATIVE" if importance < 0 else "NEUTRAL"

            if save_audio:
                # Normalization to prevent clipping
                if np.max(np.abs(y_band)) > 0:
                    y_band = y_band / np.max(np.abs(y_band)) * 0.99

                out_path = component_dir / (
                    f"{file_name}__{component}__{int(low)}-{int(high)}Hz_{importance_type}_"
                    f"{importance:+.3f}.wav"
                )
                sf.write(str(out_path), y_band, self.sr)
            
            metadata["bands"].append({
                "component": component,
                "low": low,
                "high": high,
                "importance": importance,
                "abs_importance": abs(importance),
                "type": importance_type
            })

        meta_path = save_dir / f"{file_name}_bands_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @timed("Computing importance for single component")
    def _compute_component_importance(
        self,
        sig: np.ndarray,
        component_name: str,
        audio_path: str,
        file_attempt: int = 0,
        max_file_retries: int = 3,
        retry_on_error: bool = True,
    ) -> Optional[FBDResult]:
        
        try:
            orig_prob = self._predict(sig)
        except Exception as pred_err:
            error_msg = (
                f"Prediction error for component {component_name}: "
                f"{type(pred_err).__name__}: {pred_err}"
            )
            print(f"[Warning] {error_msg}")

            if file_attempt < max_file_retries - 1 and retry_on_error:
                print(
                    f"[Info] Retrying file {audio_path} "
                    f"(attempt {file_attempt + 2}/{max_file_retries})"
                )
                time.sleep(2.0 * (file_attempt + 1))
                return None  # pozwól nadrzędnej pętli spróbować jeszcze raz

            if self.checkpoint:
                self.checkpoint.mark_as_processed(audio_path, success=False, error_msg=error_msg)
            return None
            
        # 2. Spektrogram
        S, S_db = self._compute_spectrogram(sig)
        mag, phase = librosa.magphase(S)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        # 3. Perturbacja pasm
        batch_importances: list[dict] = []
        importance_map = np.zeros_like(mag, dtype=float)

        for (low, high) in self.bands:
            trans = self._band_transition_width(low, high)
            keep = smooth_band_keep_mask(freqs, low, high, trans=trans)  # 1 outside band, 0 inside
            keep_band = keep + self.attenuation * (1.0 - keep)

            mag_p = mag * keep_band[:, None]
            S_p = mag_p * phase
            y_p = self._invert_spectrogram(S_p)

            if self.normalize_loudness:
                y_p = match_rms(sig, y_p)

            try:
                pert_prob = self._predict(y_p)
            except Exception as pred_err:
                error_msg = (
                    f"Perturbation prediction error for {component_name} "
                    f"band {low}-{high}Hz: {type(pred_err).__name__}: {pred_err}"
                )
                print(f"[Warning] {error_msg}")

                if file_attempt < max_file_retries - 1 and retry_on_error:
                    print(
                        f"[Info] Retrying file {audio_path} "
                        f"(attempt {file_attempt + 2}/{max_file_retries})"
                    )
                    time.sleep(2.0 * (file_attempt + 1))
                    return None

                if self.checkpoint:
                    self.checkpoint.mark_as_processed(audio_path, success=False, error_msg=error_msg)
                return None

            delta = float(orig_prob - pert_prob)

            batch_importances.append(
                {
                    "component": component_name,
                    "low": float(low),
                    "high": float(high),
                    "importance": float(delta),
                }
            )

            band_mask = (freqs >= low) & (freqs <= high)
            importance_map[band_mask, :] += delta

        return FBDResult(
            importance_map=importance_map,
            spectrogram_db=S_db,
            baseline_pred=orig_prob,
            y=sig,
            S=S,
            batch_importances=batch_importances,
        )

    @timed("Computing importance for bands")
    def _compute_importance(
            self, 
            audio_path: str,
            file_attempt: int = 0,
            max_file_retries: int = 3,
            retry_on_error: bool = True
        ) -> list[FBDResult]:

        y, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration, mono=True)

        components = self._separate_sources(y)
        target_names = [nm for nm in list(components.keys()) if nm in self.separation_targets]
        
        if not target_names:
            target_names = list(components.keys())

        results: list[FBDResult] = []

        for name in target_names:
            sig = components[name]
            comp_result = self._compute_component_importance(
                sig=sig,
                component_name=name,
                audio_path=audio_path,
                max_file_retries=max_file_retries,
                retry_on_error=retry_on_error
            )

            if comp_result is not None:
                results.append(comp_result)

        return results

    @tf_retry_decorator(max_retries=20)
    @timed("Processing audio file")
    def process_audio_file(
        self, 
        audio_path: str, 
        output_dir: Path,
        folder_name: str = "",
        retry_on_error: bool = True, 
        max_file_retries: int = 5) -> Optional[Dict[str, Any]]:
        """
        Process a single audio file with optional retries on error.
        
        Args:
            audio_path: Path to the audio file
            retry_on_error: Whether to retry on error
            output_dir: Directory to save the output
            folder_name: Name of the folder containing the audio file
            max_file_retries: Maximum number of attempts for a single file
        
        Returns:
            DataFrame with results or empty DataFrame in case of error
        """

        if self.profiler:
            self.profiler.reset_sample()

        file_name = Path(audio_path).stem

        if self.checkpoint:
            processed = self.checkpoint.load_processed_files()
            if str(audio_path) in processed:
                print(f"    ⏭️  Already processed, skipping...")
                return None
        
        for file_attempt in range(max_file_retries):
            try: 
                result_list = self._compute_importance(
                    audio_path=audio_path,
                    file_attempt=file_attempt,
                    max_file_retries=max_file_retries,
                    retry_on_error=retry_on_error
                )

                if not result_list:
                    if self.checkpoint:
                        self.checkpoint.mark_as_processed(audio_path, success=False, error_msg="No importance values computed") 
                    return None

                if folder_name:
                    model_output_dir = output_dir / folder_name
                    model_output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    model_output_dir = output_dir

                track_output_dir = model_output_dir / file_name
                track_output_dir.mkdir(parents=True, exist_ok=True)
                
                comp_importance_maps: dict[str, list[np.ndarray]] = defaultdict(list)
                comp_baselines: dict[str, list[float]] = defaultdict(list)
                comp_bands: dict[str, list[dict]] = defaultdict(list)

                for comp_result in result_list:
                    y_comp = comp_result.y
                    S_comp = comp_result.S
                    batch_imp = comp_result.batch_importances or []
                    component = batch_imp[0].get("component", "mixture") if batch_imp else "mixture"

                    comp_baselines[component].append(comp_result.baseline_pred)
                    comp_importance_maps[component].append(comp_result.importance_map)
                    comp_bands[component].extend(batch_imp)

                    comp_output_dir = track_output_dir / component
                    comp_output_dir.mkdir(parents=True, exist_ok=True)

                    self._save_frequency_band_importances(
                        y=y_comp,
                        S=S_comp,
                        batch_importances=batch_imp,
                        file_name=file_name,
                        save_dir=comp_output_dir
                    )
                    
                    visualize_fbp_saliency(
                        importance_map=comp_result.importance_map,
                        S=S_comp,
                        output_path=str(comp_output_dir / f"fbp_saliency_{file_name}.png"),
                        title=f"{file_name} | FBP | Pred: {comp_result.baseline_pred:.3f}",
                        sr=self.sr,
                        hop_length=self.hop_length,
                        highlight_percent=20.0,
                        abs_threshold=None,
                    )
                    

                all_batch_importances = [
                    b for bands in comp_bands.values() for b in bands
                ]    
                visualize_file_bands(
                    bands=all_batch_importances,
                    file_name=file_name,
                    folder=folder_name,
                    output_dir=track_output_dir
                )

                if self.checkpoint:
                    self.checkpoint.mark_as_processed(audio_path, success=True)

                if self.profiler:
                    self.profiler.print_sample_summary()

                components_summary = {}
                for comp, maps in comp_importance_maps.items():
                    imp_sum = np.sum(maps, axis=0)
                    components_summary[comp] = {
                        "baseline_pred_mean": float(np.mean(comp_baselines[comp])),
                        "mean_importance": float(imp_sum.mean()),
                        "max_importance": float(imp_sum.max()),
                        "min_importance": float(imp_sum.min()),
                        "std_importance": float(imp_sum.std()),
                    }

                if comp_importance_maps:
                    global_map = np.sum(
                        [np.sum(maps, axis=0) for maps in comp_importance_maps.values()],
                        axis=0,
                    )
                else:
                    global_map = result_list[0].importance_map

                return {
                    "file_path": str(audio_path),
                    "file_name": file_name,
                    "folder": folder_name,
                    "components": components_summary,
                    "global_mean_importance": float(global_map.mean()),
                    "global_max_importance": float(global_map.max()),
                    "global_min_importance": float(global_map.min()),
                    "global_std_importance": float(global_map.std()),
                    # "bands": all_batch_importances,
                }
            
            except Exception as e:
                import traceback
                print("\n--- FULL TRACEBACK ---")
                traceback.print_exc()
                print("--- END TRACEBACK ---\n")

                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[Error] Failed to process {audio_path}: {error_msg}")
                if file_attempt < max_file_retries - 1 and retry_on_error:
                    print(f"[Info] Retrying entire file (attempt {file_attempt + 2}/{max_file_retries})")
                    try:
                        import tensorflow as tf
                        import gc
                        tf.keras.backend.clear_session()
                        gc.collect()
                    except Exception:
                        pass
                    
                    continue
                else:
                    if self.checkpoint:
                        self.checkpoint.mark_as_processed(audio_path, success=False, error_msg=error_msg)

                    return None
        return None

    def run_experiment(
        self, 
        base_path: str | Path, 
        output_dir: str | Path,
        models_to_process: Optional[list] = None,
        max_samples_per_model: Optional[int] = None,
        results_path: Optional[str | Path] = None
    ) -> pd.DataFrame:
        """
        Run experiment with checkpointing.
        """

        base_path = Path(base_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if results_path is None:
            results_path = output_dir / "FBP_results.json"
        results_path = Path(results_path)

        bands_dir = output_dir / "bands"
        bands_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("🔬 Frequency Band Perturbation Experiment")
        print("=" * 70)
        print(f"📁 Dataset: {base_path}")
        print(f"📊 Output: {output_dir}")
        print(f"🎛️  Bands: {bands_dir}")
        print(f"💾 Checkpoint: {'Enabled' if self.checkpoint else 'Disabled'}")

        tmp_file = output_dir / "FBP_results_progress.csv"
        prev_results = []
        if os.path.exists(tmp_file):
            prev_results = pd.read_csv(tmp_file).to_dict("records")

        results = prev_results

        tmp_save_freq = 1
        tmp_file = output_dir / "FBP_results_progress.csv"
        
        try:
            for folder in sorted(base_path.iterdir()):
                if not folder.is_dir():
                    continue

                if models_to_process and folder.name not in models_to_process:
                    continue

                print(f"\n📁 Processing folder: {folder.name}")

                audio_files = sorted(list(folder.glob("*.mp3")) + list(folder.glob("*.wav")))

                if max_samples_per_model:
                    audio_files = audio_files[:max_samples_per_model]

                print(f"   Found {len(audio_files)} files")

                for idx, audio_file in enumerate(audio_files, 1):
                    print(f"\n  🎵 [{idx}/{len(audio_files)}] {audio_file.name}")
                    
                    result = self.process_audio_file(
                        audio_path=str(audio_file), 
                        output_dir=bands_dir,
                        folder_name = folder.name,
                        retry_on_error=True, 
                        max_file_retries=5
                    )

                    if result:
                        results.append(result)

                        if results_path:
                            model_name = result['folder']
                            file_key = result['file_name']

                            wrapper = {
                                model_name: {
                                    file_key: result
                                }
                            }

                            append_update_fbp_results(
                                new_results=wrapper,
                                results_path=results_path,
                            )
                        if len(results) % tmp_save_freq == 0:
                                pd.DataFrame(results).to_csv(tmp_file, index=False)
                                print(f"🔄 Auto-saved progress to {tmp_file}")

            if not results:
                print("\n⚠️  No results to return!")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)

            csv_path = output_dir / f"fbp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)

            print("\n" + "=" * 70)
            print("✅ Experiment completed!")
            print("=" * 70)
            print(f"📊 Processed files: {len(df)}")
            print(f"📄 Results saved: {csv_path}")
            print(f"🎛️  Bands: {bands_dir}")
            print("=" * 70 + "\n")

            if self.profiler:
                self.profiler.print_global_summary()

            return df
            
        except Exception as e:
            print(f"\n\n❌ Critical error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            if results:
                pd.DataFrame(results).to_csv(tmp_file, index=False)
                print(f"⚠️  Crash! Progress auto-saved to {tmp_file}")
            raise

    def expand_band_level_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in results_df.iterrows():
            bands = row.get('bands', None)
            if not bands:
                continue
            for b in bands:
                low = float(b['low'])
                high = float(b['high'])
                rows.append({
                    'file_path': row['file_path'],
                    'file_name': row['file_name'],
                    'folder': row['folder'],
                    'component': b.get('component', 'mixture'),
                    'low': low,
                    'high': high,
                    'band': f'{int(low)}-{int(high)}Hz',
                    'delta': float(b['importance']),
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def visualize_results(self, results_df: pd.DataFrame, output_dir: str | Path = "fbp_results") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1) WIZUALIZACJE PER PLIK (agregaty)
        if {'folder', 'mean_importance'}.issubset(results_df.columns):
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=results_df,
                x='folder',
                y='mean_importance'
            )
            plt.title("Rozkład średniej ważności pasm per model (folder)")
            plt.xlabel("Model (folder)")
            plt.ylabel("Mean importance (Δ)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(out / "boxplot_mean_importance_by_folder.png", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 6))
            sns.violinplot(
                data=results_df,
                x='folder',
                y='baseline_pred',
                inner='quartile'
            )
            plt.title("Rozkład predykcji bazowej per model (folder)")
            plt.xlabel("Model (folder)")
            plt.ylabel("Baseline prediction")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(out / "violin_baseline_pred_by_folder.png", dpi=300)
            plt.close()

        # 2) WIZUALIZACJE PER PASMO (z batch_importances)
        band_df = self.expand_band_level_results(results_df)
        if band_df.empty:
            print("⚠️ Brak danych pasmowych (bands) do wizualizacji.")
            return

        band_order = sorted(
            band_df["band"].unique(),
            key=lambda x: int(str(x).split("-")[0])
        )

        # 2a) Boxplot: Δ per band, z podziałem na komponent
        plt.figure(figsize=(14, 7))
        sns.boxplot(
            data=band_df,
            x="band",
            y="delta",
            hue="component"
        )
        plt.title("Rozkład Δ (importance) dla pasm i komponentów")
        plt.xlabel("Band (Hz)")
        plt.ylabel("Change in probability (Δ)")
        plt.xticks(rotation=45)
        plt.legend(title="Component")
        plt.tight_layout()
        plt.savefig(out / "boxplot_delta_by_band_component.png", dpi=300)
        plt.close()

        # 2b) Heatmap: średnia Δ per model (folder) i band
        pivot_model_band = band_df.pivot_table(
            index="folder",
            columns="band",
            values="delta",
            aggfunc="mean"
        ).reindex(columns=band_order)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_model_band,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"label": "Mean Δ"}
        )
        plt.title("Średnia Δ per model (folder) i pasmo")
        plt.xlabel("Band (Hz)")
        plt.ylabel("Model (folder)")
        plt.tight_layout()
        plt.savefig(out / "heatmap_delta_by_folder_band.png", dpi=300)
        plt.close()

        # 2c) Heatmap: średnia Δ per komponent i band
        pivot_comp_band = band_df.pivot_table(
            index="component",
            columns="band",
            values="delta",
            aggfunc="mean"
        ).reindex(columns=band_order)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_comp_band,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            cbar_kws={"label": "Mean Δ"}
        )
        plt.title("Średnia Δ per komponent i pasmo")
        plt.xlabel("Band (Hz)")
        plt.ylabel("Component")
        plt.tight_layout()
        plt.savefig(out / "heatmap_delta_by_component_band.png", dpi=300)
        plt.close()

        # 2d) Wykres słupkowy: mean Δ per band z podziałem na model
        grouped = band_df.groupby(["band", "folder"])["delta"].mean().reset_index()
        grouped = grouped.pivot(index="band", columns="folder", values="delta").reindex(band_order)

        grouped.plot(kind="bar", figsize=(14, 8), width=0.8)
        plt.title("Średnia Δ per pasmo z podziałem na model (folder)")
        plt.xlabel("Band (Hz)")
        plt.ylabel("Mean change in probability (Δ)")
        plt.xticks(rotation=45)
        plt.legend(title="Model (folder)", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "bar_mean_delta_by_band_and_folder.png", dpi=300)
        plt.close()

def visualize_fbp_saliency(
    importance_map: np.ndarray,
    S: np.ndarray,
    output_path: str,
    title: str,
    sr: int,
    hop_length: int,
    highlight_percent: float = 20.0,
    abs_threshold: float | None = None,
) -> None:
    """
    Wizualizacja importance_map dla FBP:
    1) oryginalny spektrogram (STFT)
    2) pełna mapa Δ
    3) tylko najbardziej istotne regiony
    4) nałożenie Δ na spektrogram
    """

    # 1) spektrogram w dB
    spectrogram_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    n_freq = S.shape[0]
    n_fft = 2 * (n_freq - 1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    y_ticks_hz = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
    y_ticks_idx = [np.argmin(np.abs(freqs - hz)) for hz in y_ticks_hz]
    y_ticks_lbl = [f"{f}" for f in y_ticks_hz]

    # maska jak w visualize_spectrogram_saliency
    if abs_threshold is not None:
        mask = np.abs(importance_map) >= abs_threshold
        maskinfo = f"|Δ pred| ≥ {abs_threshold:.2f}"
    else:
        pos_thr = np.percentile(importance_map, 100 - highlight_percent)
        neg_thr = np.percentile(importance_map, highlight_percent)
        mask = (importance_map >= pos_thr) | (importance_map <= neg_thr)
        maskinfo = f"Top ±{highlight_percent:.0f}%"

    filtered_map = np.full_like(importance_map, np.nan)
    filtered_map[mask] = importance_map[mask]

    alpha_mask = np.zeros_like(importance_map, dtype=float) + 0.25
    alpha_mask[mask] = 1.0

    fig, axes = plt.subplots(4, 1, figsize=(18, 16))

    # 1. Oryginalny spektrogram (czas–częstotliwość)
    img1 = librosa.display.specshow(
        spectrogram_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=axes[0],
        cmap="viridis",
    )
    axes[0].set_title("Original STFT Spectrogram", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Frequency (Hz)", fontsize=11)
    plt.colorbar(img1, ax=axes[0], format="%+2.0f dB")

    # 2. Pełna mapa Δ
    fullmap_absmax = np.max(np.abs(importance_map))
    im2 = axes[1].imshow(
        importance_map,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        interpolation="none",
        vmin=-fullmap_absmax,
        vmax=fullmap_absmax,
    )
    axes[1].set_title("Full Importance (Δ Prediction)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Frequency (Hz)", fontsize=11)
    axes[1].set_yticks(y_ticks_idx)
    axes[1].set_yticklabels(y_ticks_lbl)
    plt.colorbar(im2, ax=axes[1], label="Importance (Δ prediction)", orientation="vertical")

    # 3. Tylko wyróżnione regiony
    im3 = axes[2].imshow(
        filtered_map,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        interpolation="none",
        vmin=-fullmap_absmax,
        vmax=fullmap_absmax,
    )
    axes[2].set_title(f"Highlighted Importance ({maskinfo})", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Frequency (Hz)", fontsize=11)
    axes[2].set_yticks(y_ticks_idx)
    axes[2].set_yticklabels(y_ticks_lbl)
    plt.colorbar(im3, ax=axes[2], label="Importance", orientation="vertical")

    # 4. Overlay: Δ na spektrogramie
    if abs_threshold is not None or highlight_percent is not None:
        is_core = mask
        alpha_mask = np.zeros_like(importance_map, dtype=float) + 0.20
        alpha_mask[is_core] = 0.65

    axes[3].imshow(
        spectrogram_db,
        aspect="auto",
        origin="lower",
        cmap="gray",
        alpha=0.92,
    )
    axes[3].imshow(
        importance_map,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        alpha=alpha_mask,
        vmin=-fullmap_absmax,
        vmax=fullmap_absmax,
        interpolation="none",
    )
    axes[3].set_title(
        f"Spectrogram + FBP saliency\nHighlighted: {maskinfo} (alpha=1 core, 0.25 background)",
        fontsize=13,
        fontweight="bold",
    )
    axes[3].set_ylabel("Frequency (Hz)", fontsize=11)
    axes[3].set_yticks(y_ticks_idx)
    axes[3].set_yticklabels(y_ticks_lbl)
    axes[3].set_xlabel("Time frame", fontsize=11)

    stats_text = (
        f"Mean: {importance_map.mean():.4f} | "
        f"Max: {importance_map.max():.4f} | "
        f"Min: {importance_map.min():.4f}\n"
        f"{maskinfo} | Highlighted: {np.sum(mask)} ({100 * np.mean(mask):.1f}%)"
    )
    axes[3].text(
        0.02,
        0.94,
        stats_text,
        transform=axes[3].transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved FBP saliency: {output_path}")

def visualize_file_bands(
    bands: list[dict],
    file_name: str,
    folder: str,
    output_dir: Path | str
) -> None:
    """
    Wizualizacja wpływu pasm dla pojedynczego pliku:
    - barplot (Δ per band)
    - opcjonalnie oddzielnie: dodatnie / ujemne Δ
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bands:
        return

    df = pd.DataFrame(bands)
    df["band"] = df.apply(lambda r: f"{int(r['low'])}-{int(r['high'])}Hz", axis=1)
    df.sort_values("low", inplace=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="band", y="importance", hue="component")
    plt.title(f"{file_name} | {folder} | Δ per band")
    plt.xlabel("Band (Hz)")
    plt.ylabel("Change in probability (Δ)")
    plt.xticks(rotation=45)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    out_path = output_dir / f"{file_name}__band_importance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
