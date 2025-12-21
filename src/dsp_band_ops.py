# src/dsp_band_ops.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import time
from typing import Iterable, List, Tuple, Optional, Dict, Any

import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from audioLIME.factorization_spleeter import SpleeterFactorization
except Exception:
    SpleeterFactorization = None

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
    
    def log_success(self, file_path: str, num_results: int):
        """Log success of processing"""
        with open(self.progress_log, 'a', encoding='utf-8') as f:
            f.write(f"[SUCCESS] {datetime.now().isoformat()} | {file_path} | {num_results} results\n")
    
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
            return pd.DataFrame() if func.__name__ == "process_audio" else None
        return wrapper
    return decorator

class Predictor:
    """
    Minimal interface required by FrequencyBandPerturbation.
    Implementations: LocalSonnics / RemoteSonnics in src/model_api.py.
    """
    def predict(self, audio_wave: np.ndarray, sr: int) -> float:
        raise NotImplementedError

@dataclass
class FBPConfig:
    model_time: int = 120
    sr: int = 44100
    use_mel: bool = False
    n_mels: int = 128
    use_separation: bool = False
    separation_model: str = "spleeter:2stems"
    separation_targets: Tuple[str, ...] = ("vocals0", "accompaniment0")
    lufs: Optional[float] = None  # if None, no LUFS normalization
    bands_preset: str = "default"
    custom_bands: Optional[List[Tuple[int, int]]] = None
    attenuation: float = 0.0  # 0 mute, 1 no change
    transition_mode: str = "rel"  # "abs" or "rel"
    transition_hz: float = 200.0
    transition_rel: float = 0.2
    transition_min_hz: float = 20.0
    transition_max_hz: float = 2000.0
    normalize_loudness: bool = True
    n_fft: int = 2048
    hop_length: int = 512

class FrequencyBandPerturbation:
    def __init__(self, predictor: Predictor, cfg: FBPConfig, checkpoint_dir: Optional[str | Path] = None):
        self.predictor = predictor
        self.cfg = cfg

        if checkpoint_dir:
            self.checkpoint = ExperimentCheckpoint(checkpoint_dir)
        else:
            self.checkpoint = None

        if cfg.custom_bands is not None:
            self.bands = list(cfg.custom_bands)
        else:
            self.bands = FREQUENCY_BAND_PRESETS.get(cfg.bands_preset, FREQUENCY_BAND_PRESETS["default"])

        self.cfg.attenuation = float(np.clip(self.cfg.attenuation, 0.0, 1.0))
        self.cfg.transition_hz = float(max(0.0, self.cfg.transition_hz))
        self.cfg.transition_rel = float(max(0.0, self.cfg.transition_rel))
        self.cfg.transition_min_hz = float(max(0.0, self.cfg.transition_min_hz))
        self.cfg.transition_max_hz = float(max(0.0, self.cfg.transition_max_hz))

    def _band_transition_width(self, low: float, high: float) -> float:
        bw = float(high - low)
        if self.cfg.transition_mode == "rel":
            trans = bw * self.cfg.transition_rel
            trans = float(np.clip(trans, self.cfg.transition_min_hz, self.cfg.transition_max_hz))
        else:
            trans = float(self.cfg.transition_hz)
        return trans

    def _predict(self, wave: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.predictor.predict(wave, self.cfg.sr))

    def _separate_sources(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.cfg.use_separation:
            return {"mixture": audio}

        if SpleeterFactorization is None:
            print("[Warning] Spleeter not available, falling back to mixture.")
            return {"mixture": audio}

        fact = SpleeterFactorization(
            input=audio,
            target_sr=self.cfg.sr,
            temporal_segmentation_params=1,
            composition_fn=None,
            model_name=self.cfg.separation_model,
        )
        return dict(zip(fact._components_names, fact.original_components))

    @tf_retry_decorator(max_retries=20)
    def process_audio(self, audio_path: str, retry_on_error: bool = True, max_file_retries: int = 5) -> pd.DataFrame:
        """
        Process a single audio file with optional retries on error.
        
        Args:
            audio_path: Path to the audio file
            retry_on_error: Whether to retry on error
            max_file_retries: Maximum number of attempts for a single file
        
        Returns:
            DataFrame with results or empty DataFrame in case of error
        """
        
        for file_attempt in range(max_file_retries):
            try: 
                results: List[Dict[str, Any]] = []

                y, _ = librosa.load(audio_path, sr=self.cfg.sr, mono=True, duration=self.cfg.model_time)
                n_fft = self.cfg.n_fft
                hop = self.cfg.hop_length

                components = self._separate_sources(y)
                target_names = [nm for nm in list(components.keys()) if nm in self.cfg.separation_targets]
                if not target_names:
                    target_names = list(components.keys())

                for name in target_names:
                    sig = components[name]
                    
                    try:
                        orig_prob = self._predict(sig)
                    except Exception as pred_err:
                        error_msg = f"Prediction error for component {name}: {type(pred_err).__name__}: {pred_err}"
                        print(f"[Warning] {error_msg}")
                    
                        if file_attempt < max_file_retries - 1 and retry_on_error:
                            print(f"[Info] Retrying file {audio_path} (attempt {file_attempt + 2}/{max_file_retries})")
                            time.sleep(2.0 * (file_attempt + 1))
                            break
                        else:
                            if self.checkpoint:
                                self.checkpoint.mark_as_processed(audio_path, success=False, error_msg=error_msg)
                            raise

                    if self.cfg.use_mel:
                        S_mel = librosa.feature.melspectrogram(
                            y=sig, sr=self.cfg.sr, n_fft=n_fft, hop_length=hop, n_mels=self.cfg.n_mels
                        )
                        _ = librosa.power_to_db(S_mel, ref=np.max)

                        S = librosa.stft(sig, n_fft=n_fft, hop_length=hop, window="hann", center=True)
                        mag, phase = librosa.magphase(S)
                        freqs = librosa.fft_frequencies(sr=self.cfg.sr, n_fft=n_fft)
                    else:
                        S = librosa.stft(sig, n_fft=n_fft, hop_length=hop, window="hann", center=True)
                        mag, phase = librosa.magphase(S)
                        freqs = librosa.fft_frequencies(sr=self.cfg.sr, n_fft=n_fft)

                    for (low, high) in self.bands:
                        trans = self._band_transition_width(low, high)
                        keep = smooth_band_keep_mask(freqs, low, high, trans=trans)  # 1 outside band, 0 inside band
                        keep_band = keep + self.cfg.attenuation * (1.0 - keep)
                        mag_p = mag * keep_band[:, None]

                        S_p = mag_p * phase
                        y_p = librosa.istft(S_p, hop_length=hop, window="hann", center=True)

                        if self.cfg.normalize_loudness:
                            y_p = match_rms(sig, y_p)

                        
                        try:
                            pert_prob = self._predict(y_p)
                        except Exception as pred_err:
                            error_msg = f"Perturbation prediction error for band {low}-{high}Hz: {type(pred_err).__name__}: {pred_err}"
                            print(f"[Warning] {error_msg}")
                        
                            if file_attempt < max_file_retries - 1 and retry_on_error:
                                print(f"[Info] Retrying file {audio_path} (attempt {file_attempt + 2}/{max_file_retries})")
                                time.sleep(2.0 * (file_attempt + 1))
                                break
                            else:
                                if self.checkpoint:
                                    self.checkpoint.mark_as_processed(audio_path, success=False, error_msg=error_msg)
                                raise
                            
                        delta = float(orig_prob - pert_prob)

                        results.append({
                            "file_path": str(audio_path),
                            "band": f"{low}-{high}Hz",
                            "original_prob": float(orig_prob),
                            "perturbed_prob": float(pert_prob),
                            "delta": delta,
                            "component": name,
                        })

                if self.checkpoint:
                    self.checkpoint.mark_as_processed(audio_path, success=True)
                    self.checkpoint.log_success(audio_path, num_results=len(results))

                return pd.DataFrame(results)
            
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"[Error] Failed to process {audio_path}: {error_msg}")
                
                if file_attempt < max_file_retries - 1 and retry_on_error:
                    print(f"[Info] Retrying entire file (attempt {file_attempt + 2}/{max_file_retries})")
                    time.sleep(2.0 * (file_attempt + 1))

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

                    return pd.DataFrame()
        return pd.DataFrame()

    def load_previous_results(self, output_dir: str | Path, experiment_name: str) -> Optional[pd.DataFrame]:
        """
        Load previous experiment results if they exist.

        Args:
            output_dir: Directory with results
            experiment_name: Experiment name

        Returns:
            DataFrame with previous results or None if they don't exist
        """
        exp_dir = Path(output_dir) / experiment_name
        
        if not exp_dir.exists():
            return None

        csv_files = sorted(exp_dir.glob("fbp_results_*.csv"))
        
        if not csv_files:
            return None

        latest_csv = csv_files[-1]
        
        try:
            df = pd.read_csv(latest_csv)
            print(f"📥 Loaded previous results: {latest_csv.name}")
            print(f"   Number of results: {len(df)}")
            return df
        except Exception as e:
            print(f"⚠️  Failed to load previous results: {e}")
            return None

    def run_experiment(
        self, 
        base_path: str | Path, 
        limit_per_folder: Optional[int] = None,
        resume: bool = True,
        force_reprocess: bool = False,
        previous_results: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run experiment with checkpointing.
        
        Args:
            base_path: Path to the data folder
            limit_per_folder: Limit files per folder
            resume: Whether to resume from the last checkpoint
            force_reprocess: Force reprocess all files (ignore checkpoint)
        """
        base = Path(base_path)
        all_rows: List[pd.DataFrame] = []
        
        if previous_results is not None and not force_reprocess:
            print(f"🔗 Connecting to previous results ({len(previous_results)} rows)")
            all_rows.append(previous_results)

        processed_files = set()
        if self.checkpoint and resume and not force_reprocess:
            processed_files = self.checkpoint.load_processed_files()
            if processed_files:
                print(f"\n📋 Resumed checkpoint: {len(processed_files)} files already processed")
                stats = self.checkpoint.get_stats()
                print(f"   Success: {stats['total_processed'] - stats['total_failed']}, Errors: {stats['total_failed']}")
        
        total_files = 0
        files_to_process = {}
        
        for folder in base.iterdir():
            if not folder.is_dir():
                continue
            audio_files = list(folder.glob("*"))
            if limit_per_folder is not None:
                audio_files = audio_files[:int(limit_per_folder)]

            if resume and not force_reprocess:
                audio_files = [f for f in audio_files if str(f) not in processed_files]
            
            files_to_process[folder] = audio_files
            total_files += len(audio_files)

        print(f"\n🔊 Files to process: {total_files}")

        processed_count = 0
        failed_count = 0
        
        for folder, audio_files in files_to_process.items():
            if not audio_files:
                continue

            print(f"\n📁 Processing folder: {folder.name} ({len(audio_files)} files)")

            for idx, f in enumerate(audio_files, 1):
                print(f"  🎵 [{idx}/{len(audio_files)}] {f.name}")
                
                df = self.process_audio(str(f), retry_on_error=True, max_file_retries=5)
                
                if not df.empty:
                    df["folder"] = folder.name
                    all_rows.append(df)
                    processed_count += 1
                    print(f"     ✅ Success ({len(df)} results)")
                else:
                    failed_count += 1
                    print(f"     ❌ Error (file skipped)")

                total_done = len(processed_files) + processed_count + failed_count
                print(f"  📊 Global progress: {total_done}/{len(processed_files) + total_files} " +
                    f"(Success: {processed_count}, Errors: {failed_count})")

        if not all_rows:
            print("\n⚠️  No results to return!")
            return pd.DataFrame()
        
        final_df = pd.concat(all_rows, ignore_index=True)

        if 'file_path' in final_df.columns and 'band' in final_df.columns:
            before_dedup = len(final_df)
            final_df = final_df.drop_duplicates(subset=['file_path', 'band', 'component'], keep='last')
            after_dedup = len(final_df)
            if before_dedup != after_dedup:
                print(f"🔄 Removed {before_dedup - after_dedup} duplicates")

        print(f"\n✅ Experiment completed!")
        print(f"   Processed files: {processed_count}")
        print(f"   Failed files: {failed_count}")
        print(f"   Total results: {len(final_df)}")

        if self.checkpoint:
            failed_files = self.checkpoint.get_failed_files()
            if failed_files:
                print(f"\n❌ Files with errors ({len(failed_files)}):")
                for failed in failed_files[-10:]:
                    print(f"   - {Path(failed['file_path']).name}: {failed['error'][:80]}")
        
        return final_df

    def save_experiment_results(
        self,
        results_df: pd.DataFrame,
        output_dir: str | Path,
        experiment_name: str = "exp",
        extra_meta: Optional[Dict[str, Any]] = None,
        is_merged: bool = False
    ) -> Dict[str, str]:

        out = Path(output_dir)
        exp_dir = out / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        suffix = "_merged" if is_merged else ""
        csv_path = exp_dir / f"fbp_results{suffix}_{ts}.csv"
        params_path = exp_dir / f"fbp_params{suffix}_{ts}.json"

        results_df.to_csv(csv_path, index=False)

        bands_with_transition = []
        for (low, high) in self.bands:
            eff_trans = self._band_transition_width(low, high)
            bands_with_transition.append({
                "low": float(low),
                "high": float(high),
                "bandwidth": float(high - low),
                "transition_effective_hz": float(eff_trans),
            })

        params: Dict[str, Any] = {
            "timestamp": ts,
            "experiment_name": experiment_name,
            "sr": self.cfg.sr,
            "model_time": self.cfg.model_time,
            "use_mel": self.cfg.use_mel,
            "n_mels": self.cfg.n_mels if self.cfg.use_mel else None,
            "use_separation": self.cfg.use_separation,
            "separation_model": self.cfg.separation_model,
            "separation_targets": list(self.cfg.separation_targets) if self.cfg.separation_targets else None,
            "bands": bands_with_transition,
            "n_fft": self.cfg.n_fft,
            "hop_length": self.cfg.hop_length,
            "rows": int(len(results_df)),
            "folders_covered": sorted(results_df["folder"].unique().tolist()) if "folder" in results_df.columns else None,
            "attenuation": self.cfg.attenuation,
            "transition_mode": self.cfg.transition_mode,
            "transition_hz": self.cfg.transition_hz,
            "transition_rel": self.cfg.transition_rel,
            "transition_min_hz": self.cfg.transition_min_hz,
            "transition_max_hz": self.cfg.transition_max_hz,
            "normalize_loudness": self.cfg.normalize_loudness,
        }
        if extra_meta:
            params["extra_meta"] = extra_meta

        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        return {
            "results_csv": str(csv_path),
            "params_json": str(params_path),
            "experiment_dir": str(exp_dir),
        }

    def visualize_results(self, results_df: pd.DataFrame, output_dir: str | Path = "fbp_results") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        band_order = sorted(results_df["band"].unique(), key=lambda x: int(str(x).split("-")[0]))

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=results_df, x="band", y="delta", hue="component")
        plt.title("Distribution of delta for band and component")
        plt.xlabel("Band (Hz)")
        plt.ylabel("Change in Probability (Δ)")
        plt.xticks(rotation=45)
        plt.legend(title="Component")
        plt.tight_layout()
        plt.savefig(out / "boxplot_delta_by_band_component.png", dpi=300)
        plt.close()

        for comp in results_df["component"].unique():
            comp_df = results_df[results_df["component"] == comp]
            pivot = comp_df.pivot_table(index="folder", columns="band", values="delta", aggfunc="mean")
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot[band_order], annot=True, fmt=".3f", cmap="coolwarm", center=0,
                        linewidths=0.5, cbar_kws={"label": "Δ"})
            plt.title(f"Heatmap: component [{comp}]")
            plt.tight_layout()
            plt.savefig(out / f"heatmap_delta_{comp}.png", dpi=300)
            plt.close()

        grouped = results_df.groupby(["component", "band"])["delta"].mean().unstack().fillna(0)
        grouped = grouped[band_order]
        grouped.T.plot(kind="bar", figsize=(14, 8), width=0.8)
        plt.title("Comparison of mean Δ (per band and component)")
        plt.xlabel("Band (Hz)")
        plt.ylabel("Mean change in probability (Δ)")
        plt.xticks(rotation=45)
        plt.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "band_impact_by_component.png", dpi=300)
        plt.close()

    def visualize_embedding(self, results_df: pd.DataFrame, output_dir: str | Path = "fbp_results", perplexity: int = 30) -> None:
        import umap
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        pivot = results_df.pivot_table(index=["file_path", "component", "folder"], columns="band",
                                       values="delta", aggfunc="mean").fillna(0)
        X = pivot.values
        labels = pivot.index.get_level_values("folder") if "folder" in pivot.index.names else None
        components = pivot.index.get_level_values("component")

        palette = sns.color_palette("Set2", n_colors=len(set(labels)) if labels is not None else 10)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, style=components, palette=palette)
        plt.title("PCA: projection of band deltas")
        plt.tight_layout()
        plt.savefig(out / "embedding_pca.png", dpi=300)
        plt.close()

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, style=components, palette=palette)
        plt.title("t-SNE: projection of band deltas")
        plt.tight_layout()
        plt.savefig(out / "embedding_tsne.png", dpi=300)
        plt.close()

        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, style=components, palette=palette)
        plt.title("UMAP: projection of band deltas")
        plt.tight_layout()
        plt.savefig(out / "embedding_umap.png", dpi=300)
        plt.close()
