# src/spectrogram_explainability.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Tuple, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import tempfile
import os

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class SpectrogramCheckpoint:
    """
        Manages checkpointing for spectrogram experiments
        Args:
            checkpoint_dir: Directory to store checkpoint and logs
    """
    
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "spectrogram_checkpoint.json"
        self.progress_log = self.checkpoint_dir / "spectrogram_progress.txt"
        
    def load_processed_files(self) -> set:
        """Load list of already processed files"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('processed_files', []))
        return set()
    
    def mark_as_processed(self, file_path: str):
        """Mark file as processed"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'processed_files': [], 'last_updated': None}
        
        if file_path not in data['processed_files']:
            data['processed_files'].append(file_path)
        
        data['last_updated'] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        with open(self.progress_log, 'a', encoding='utf-8') as f:
            f.write(f"[PROCESSED] {datetime.now().isoformat()} | {file_path}\n")

def _save_windows_for_group(
    y: np.ndarray,
    S: np.ndarray,
    patch_importances: list[dict],
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
    top_n: int,
    base_save_dir: Path,
    file_name: str,
    use_original_audio: bool,
    group_name: str,
    sort_reverse: bool
):
    save_dir = base_save_dir / group_name
    save_dir.mkdir(parents=True, exist_ok=True)

    sorted_patches = sorted(
        patch_importances,
        key=lambda p: abs(p["importance"]),
        reverse=sort_reverse
    )

    top_patches = sorted_patches[:top_n]

    metadata = {
        "file_name": file_name,
        "group": group_name,
        "top_n": int(len(top_patches)),
        "windows": []
    }

    for rank, p in enumerate(top_patches, 1):
        t_start = p["t_start"]
        t_end = p["t_end"]
        f_start = p["f_start"]
        f_end = p["f_end"]
        importance = float(p["importance"])
        abs_importance = float(abs(importance))

        window_frames = t_end - t_start
        window_samples = max(1, window_frames * hop_length)

        masked_S = np.zeros_like(S)
        masked_S[f_start:f_end, t_start:t_end] = S[f_start:f_end, t_start:t_end]

        if use_original_audio:
            start_sample = int(t_start * hop_length)
            end_sample = int(start_sample + window_samples)
            end_sample = min(end_sample, len(y))

            y_window = y[start_sample:end_sample]

            if len(y_window) < window_samples:
                y_window = np.pad(y_window, (0, window_samples - len(y_window)))

        else:
            masked_S = np.zeros_like(S)
            masked_S[f_start:f_end, t_start:t_end] = S[f_start:f_end, t_start:t_end]

            y_window_full = librosa.feature.inverse.mel_to_audio(
                masked_S,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_iter=n_iter
            )

            start_sample = int(t_start * hop_length)
            start_sample = max(0, start_sample)
            end_sample = int(start_sample + window_samples)
            end_sample = min(end_sample, len(y_window_full))

            y_window = y_window_full[start_sample:end_sample]
            if len(y_window_full) < window_samples:
                y_window_full = np.pad(y_window_full, (0, window_samples - len(y_window_full)))

        importance_type = "POSITIVE" if importance > 0 else "NEGATIVE" if importance < 0 else "NEUTRAL"

        out_path = base_save_dir / (
            f"{file_name}__{group_name}{rank}_patch_{importance_type}_"
            f"{abs_importance:.3f}_t{t_start}-{t_end}_f{f_start}-{f_end}.wav"
        )
        sf.write(str(out_path), y_window, sr)

        metadata["windows"].append({
            "rank": int(rank),
            "t_start": int(t_start),
            "t_end": int(t_end),
            "f_start": int(f_start),
            "f_end": int(f_end),
            "start_time_sec": float(t_start * hop_length / sr),
            "end_time_sec": float(t_end * hop_length / sr),
            "importance": importance,
            "abs_importance": abs_importance,
            "type": importance_type
        })

    meta_path = base_save_dir / f"{file_name}__{group_name}_occlusion_patches_from_list.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def save_top_occlusion_patches_from_list(
    y: np.ndarray,
    S: np.ndarray,
    patch_importances: list[dict],
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
    top_n: int,
    save_dir: Path | str,
    file_name: str,
    use_original_audio: bool = True
):
    base_save_dir = Path(save_dir)
    base_save_dir.mkdir(parents=True, exist_ok=True)

    _save_windows_for_group(
        y=y,
        S=S,
        patch_importances=patch_importances,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_iter=n_iter,
        top_n=top_n,
        base_save_dir=base_save_dir,
        file_name=file_name,
        use_original_audio=use_original_audio,
        group_name="best",
        sort_reverse=True
    )

    _save_windows_for_group(
        y=y,
        S=S,
        patch_importances=patch_importances,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_iter=n_iter,
        top_n=top_n,
        base_save_dir=base_save_dir,
        file_name=file_name,
        use_original_audio=use_original_audio,
        group_name="worst",
        sort_reverse=False
    )

def compute_occlusion_map(
    audio_path: str,
    predict_fn: Callable,
    sr: int = 44100,
    duration: int = 120,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    n_mels: int = 128,
    n_iter: int = 256,
    patch_size: Tuple[int, int] = (16, 16),
    stride: Tuple[int, int] = (8, 8),
    occlusion_value: float = 0.0,
    baseline_threshold: float = 0.3,
    verbose: bool = True
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    Generate occlusion-based saliency map for audio spectrogram.
    
    :param audio_path: Path to audio file
    :param predict_fn: Function that takes waveform (np.array) and sr, returns fake probability
    :param sr: Sample rate
    :param duration: Duration in seconds
    :param n_fft: FFT size
    :param hop_length: Hop length
    :param win_length: Window length
    :param n_mels: Number of mel bands
    :param patch_size: Size of occlusion patch (time_frames, freq_bins)
    :param stride: Stride for sliding window
    :param occlusion_value: Value to use for occlusion (0.0 = silence)
    :param baseline_threshold: Only compute if baseline pred > threshold
    :param verbose: Print progress
    :return: (importance_map, spectrogram_db, baseline_prediction)
    """
    
    y, _ = librosa.load(audio_path, sr=sr, duration=duration, mono=True)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, 
        hop_length=hop_length, win_length=win_length, fmax=sr//2
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    baseline_pred = predict_fn(y, sr)
    if verbose:
        print(f"    Baseline prediction: {baseline_pred:.4f}")
    
    if baseline_pred < baseline_threshold:
        if verbose:
            print(f"    ⏭️  Baseline too low ({baseline_pred:.4f}), skipping...")
        return None, S_db, baseline_pred
    
    n_freq, n_time = S.shape
    importance_map = np.zeros((n_freq, n_time))
    count_map = np.zeros((n_freq, n_time))
    
    t_patches = max(1, (n_time - patch_size[0]) // stride[0] + 1)
    f_patches = max(1, (n_freq - patch_size[1]) // stride[1] + 1)
    total_patches = t_patches * f_patches
    
    if verbose:
        print(f"    Processing {total_patches} patches (patch_size={patch_size}, stride={stride})...")
    
    patch_positions = []
    for t_start in range(0, n_time - patch_size[0] + 1, stride[0]):
        for f_start in range(0, n_freq - patch_size[1] + 1, stride[1]):
            patch_positions.append((t_start, f_start))
    
    pbar = tqdm(
        patch_positions,
        desc=f"    {Path(audio_path).stem[:20]}",
        unit="patch",
        ncols=120,
        disable=not verbose,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    importance_values = []
    patch_importances = []

    for t_start, f_start in pbar:
        S_occluded = S.copy()
        
        t_end = min(t_start + patch_size[0], n_time)
        f_end = min(f_start + patch_size[1], n_freq)
        
        S_occluded[f_start:f_end, t_start:t_end] = occlusion_value
        
        y_occluded = librosa.feature.inverse.mel_to_audio(
            S_occluded, 
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length,
            n_iter=n_iter
        )
        
        if len(y_occluded) > len(y):
            y_occluded = y_occluded[:len(y)]
        elif len(y_occluded) < len(y):
            y_occluded = np.pad(y_occluded, (0, len(y) - len(y_occluded)))
        
        occluded_pred = predict_fn(y_occluded, sr)
        
        importance = baseline_pred - occluded_pred
        importance_values.append(importance)

        patch_importances.append({
            "t_start": int(t_start),
            "t_end": int(t_end),
            "f_start": int(f_start),
            "f_end": int(f_end),
            "importance": importance
        })
        
        importance_map[f_start:f_end, t_start:t_end] += importance
        count_map[f_start:f_end, t_start:t_end] += 1
        
        if len(importance_values) > 0:
            mean_imp = np.mean(importance_values[-10:])
            pbar.set_postfix({
                'imp': f'{importance:+.3f}',
                'avg': f'{mean_imp:+.3f}'
            })
    
    pbar.close()
    
    importance_map = importance_map / (count_map + 1e-8)
    
    if verbose:
        print(f"    ✅ Completed | Mean importance: {importance_map.mean():.4f}, "
              f"Max: {importance_map.max():.4f}")
    
    return importance_map, S_db, baseline_pred, patch_importances

def compute_rise_map(
    audio_path: str,
    predict_fn: Callable,
    sr: int = 44100,
    duration: int = 120,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    n_mels: int = 128,
    n_iter: int = 256,
    n_masks: int = 500,
    mask_probability: float = 0.5,
    mask_size: float = 0.1,
    baseline_threshold: float = 0.3,
    verbose: bool = True
) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
    """
    RISE (Randomized Input Sampling for Explanation).
    Much faster than occlusion - uses random masks instead of sliding window.
    
    :param audio_path: Path to audio file
    :param predict_fn: Function that takes waveform and sr, returns fake probability
    :param sr: Sample rate
    :param duration: Duration in seconds
    :param n_fft: FFT size
    :param hop_length: Hop length
    :param win_length: Window length
    :param n_mels: Number of mel bands
    :param n_masks: Number of random masks (200-500 recommended)
    :param mask_probability: Probability of keeping each cell (0.5 = 50% masked)
    :param mask_size: NOT USED (kept for compatibility)
    :param baseline_threshold: Minimum baseline prediction to process
    :param verbose: Print progress
    :return: (importance_map, spectrogram_db, baseline_prediction)
    """
    
    y, _ = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length, fmax=sr//2
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    baseline_pred = predict_fn(y, sr)
    if verbose:
        print(f"    Baseline prediction: {baseline_pred:.4f}")
    
    if baseline_pred < baseline_threshold:
        if verbose:
            print(f"    ⏭️  Baseline too low ({baseline_pred:.4f}), skipping...")
        return None, S_db, baseline_pred
    
    n_freq, n_time = S.shape
    
    if verbose:
        print(f"    Processing {n_masks} random masks (RISE method)...")
        print(f"    Spectrogram shape: {n_freq} freq × {n_time} time")
    
    importance_map = np.zeros((n_freq, n_time))
    
    pbar = tqdm(
        range(n_masks),
        desc="    Masks",
        unit="mask",
        ncols=100,
        disable=not verbose,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    predictions = []
    
    for mask_idx in pbar:
        # Generate random binary mask
        mask = (np.random.rand(n_freq, n_time) > (1 - mask_probability)).astype(float)
        S_masked = S * mask
        
        # Reconstruct audio
        y_masked = librosa.feature.inverse.mel_to_audio(
            S_masked, 
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length,
            win_length=win_length,
            n_iter=n_iter
        )
        
        if len(y_masked) > len(y):
            y_masked = y_masked[:len(y)]
        elif len(y_masked) < len(y):
            y_masked = np.pad(y_masked, (0, len(y) - len(y_masked)))
        
        masked_pred = predict_fn(y_masked, sr)
        predictions.append(masked_pred)
        
        # Accumulate: importance = mask × prediction
        # High prediction with this mask -> these regions are important
        importance_map += mask * masked_pred
        
        if len(predictions) > 0:
            recent_avg = np.mean(predictions[-10:])
            pbar.set_postfix({
                'pred': f'{masked_pred:.3f}',
                'avg': f'{recent_avg:.3f}'
            })
    
    pbar.close()
    
    # Normalize by number of times each cell was unmasked
    # Expected number of masks per cell = n_masks * mask_probability
    importance_map = importance_map / (n_masks * mask_probability + 1e-8)
    
    # Normalize to [0, 1] range
    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
    
    if verbose:
        print(f"    ✅ Completed | Mean importance: {importance_map.mean():.4f}, "
              f"Max: {importance_map.max():.4f}")
    
    return importance_map, S_db, baseline_pred

def visualize_spectrogram_saliency(
    importance_map: np.ndarray,
    spectrogram_db: np.ndarray,
    output_path: str,
    title: str = "Spectrogram Saliency Map",
    sr: int = 44100,
    hop_length: int = 512,
    highlight_percent: float = 20.0,
    abs_threshold: float = None
):
    """
    Visualize full heatmap, masked version (top ±percentile or abs thresh)
    + overlay with alpha (soft shading of less important regions).
    """

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

    # 1. Original spectrogram
    img1 = librosa.display.specshow(
        spectrogram_db, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='mel', ax=axes[0], cmap='viridis'
    )
    axes[0].set_title('Original Mel Spectrogram', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Frequency (Hz)', fontsize=11)
    plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    # 2. Full heatmap (centered colormap)
    fullmap_absmax = np.max(np.abs(importance_map))
    im2 = axes[1].imshow(
        importance_map, aspect='auto', origin='lower',
        cmap='seismic', interpolation='none',
        vmin=-fullmap_absmax, vmax=fullmap_absmax
    )
    axes[1].set_title('Full Importance (Δ Prediction)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Mel Bin', fontsize=11)
    plt.colorbar(im2, ax=axes[1], label='Importance (Δ prediction)', orientation='vertical')

    # 3. Masked heatmap
    im3 = axes[2].imshow(
        filtered_map, aspect='auto', origin='lower',
        cmap='seismic', interpolation='none',
        vmin=-fullmap_absmax, vmax=fullmap_absmax
    )
    axes[2].set_title(f'Highlighted Importance ({maskinfo})', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Mel Bin', fontsize=11)
    plt.colorbar(im3, ax=axes[2], label='Importance', orientation='vertical')

    # 4. Overlay: alpha soft shading
    # Rescale importance_map to [0, 1] for alpha (optionally: only for selected core)
    scaled_alpha = 0.20 + 0.55 * (np.abs(importance_map) / np.max(np.abs(importance_map)))  # 0.2 background, up to 0.75

    if abs_threshold or highlight_percent:
        is_core = mask
        alpha_mask = np.zeros_like(importance_map, dtype=float) + 0.20
        alpha_mask[is_core] = 0.65
    else:
        alpha_mask = scaled_alpha

    axes[3].imshow(
        spectrogram_db, aspect='auto', origin='lower', cmap='gray', alpha=0.92
    )
    axes[3].imshow(
        importance_map, aspect='auto', origin='lower',
        cmap='seismic', alpha=alpha_mask,
        vmin=-fullmap_absmax, vmax=fullmap_absmax, interpolation='none'
    )
    axes[3].set_title(f'Spectrogram + Saliency\nHighlighted: {maskinfo} (alpha=1 core, 0.25 tło)', fontsize=13, fontweight='bold')
    axes[3].set_ylabel('Mel Bin', fontsize=11)
    axes[3].set_xlabel('Time Frame', fontsize=11)

    # Statistics
    stats_text = f"Mean: {importance_map.mean():.4f} | Max: {importance_map.max():.4f} | Min: {importance_map.min():.4f}\n"
    stats_text += f"{maskinfo} | Highlighted: {np.sum(mask)} ({100*np.mean(mask):.1f}%)"
    axes[3].text(0.02, 0.94, stats_text, transform=axes[3].transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")

class SpectrogramExplainability:
    """Main class for spectrogram-based explainability experiments"""
    
    def __init__(
        self,
        predictor,
        sr: int = 44100,
        duration: int = 120,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        n_mels: int = 128,
        n_iter: int = 256,
        method: str = "rise",  # "rise" or "occlusion"
        use_original_audio: bool = True,
        patch_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (8, 8),
        n_masks: int = 500,
        mask_probability: float = 0.5,
        checkpoint_dir: Optional[str | Path] = None,
        highlight_percent: float = 20.0,
        abs_threshold: float = 0.0
    ):
        self.predictor = predictor
        self.sr = sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_iter = n_iter
        self.method = method.lower()
        
        # Occlusion params
        self.patch_size = patch_size
        self.stride = stride
        self.use_original_audio = use_original_audio
        
        # RISE params
        self.n_masks = n_masks
        self.mask_probability = mask_probability

        # visualization params
        self.highlight_percent = highlight_percent
        self.abs_threshold = abs_threshold
        
        self.checkpoint = None
        if checkpoint_dir:
            self.checkpoint = SpectrogramCheckpoint(checkpoint_dir)
    
    def _predict_fn(self, waveform: np.ndarray, sr: int) -> float:
        """Wrapper for predictor"""
        try: 
            return float(self.predictor.predict(waveform, sr))
        except Exception as e:
            print(f"⚠️  Predict CRASH: {type(e).__name__}: {e}")
            print("Using SAFE FALLBACK: 0.0")
            return 0.0
    
    def process_audio_file(
        self,
        audio_path: str,
        output_dir: Path,
        baseline_threshold: float = 0.3,
        folder_name: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Process single audio file.
        
        :param audio_path: Path to audio file
        :param output_dir: Output directory for visualizations
        :param baseline_threshold: Minimum baseline prediction to process
        :return: Dictionary with results or None if skipped
        """
        
        file_name = Path(audio_path).stem
        
        if self.checkpoint:
            processed = self.checkpoint.load_processed_files()
            if str(audio_path) in processed:
                print(f"    ⏭️  Already processed, skipping...")
                return None
        
        if self.method == "rise":
            importance_map, spectrogram_db, baseline_pred = compute_rise_map(
                audio_path=audio_path,
                predict_fn=self._predict_fn,
                sr=self.sr,
                duration=self.duration,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                n_iter=self.n_iter,
                n_masks=self.n_masks,
                mask_probability=self.mask_probability,
                baseline_threshold=baseline_threshold,
                verbose=True
            )
        else:
            importance_map, spectrogram_db, baseline_pred, patch_importances = compute_occlusion_map(
                audio_path=audio_path,
                predict_fn=self._predict_fn,
                sr=self.sr,
                duration=self.duration,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                n_iter=self.n_iter,
                patch_size=self.patch_size,
                stride=self.stride,
                baseline_threshold=baseline_threshold,
                verbose=True
            )
        
        if importance_map is None:
            if self.checkpoint:
                self.checkpoint.mark_as_processed(str(audio_path))
            return None
        
        if folder_name:
            model_output_dir = output_dir / folder_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            model_output_dir = output_dir

        track_output_dir = model_output_dir / file_name
        track_output_dir.mkdir(parents=True, exist_ok=True)

        output_path = track_output_dir / f"saliency_{file_name}.png"
        method_name = "RISE" if self.method == "rise" else "Occlusion"
        
        visualize_spectrogram_saliency(
            importance_map=importance_map,
            spectrogram_db=spectrogram_db,
            output_path=str(output_path),
            title=f"{file_name} | {method_name} | Pred: {baseline_pred:.3f}",
            sr=self.sr,
            highlight_percent=self.highlight_percent,
            abs_threshold=self.abs_threshold
        )

        if importance_map is not None and self.method == 'occlusion':
            y, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration, mono=True)
            S = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft,
                hop_length=self.hop_length, win_length=self.win_length, fmax=self.sr // 2
            )

            windows_dir = track_output_dir / "top_windows"
            windows_dir.mkdir(exist_ok=True)

            save_top_occlusion_patches_from_list(
                y=y,
                S=S,
                patch_importances=patch_importances,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_iter=self.n_iter,
                top_n=5,
                save_dir=windows_dir,
                file_name=file_name,
                use_original_audio=self.use_original_audio
            )
        else:
            pass

        if self.checkpoint:
            self.checkpoint.mark_as_processed(str(audio_path))
        
        return {
            'file_path': str(audio_path),
            'file_name': file_name,
            'folder': folder_name,
            'method': self.method,
            'baseline_pred': float(baseline_pred),
            'mean_importance': float(importance_map.mean()),
            'max_importance': float(importance_map.max()),
            'min_importance': float(importance_map.min()),
            'std_importance': float(importance_map.std()),
            'p90_importance': float(np.percentile(importance_map, 90)),
            'p10_importance': float(np.percentile(importance_map, 10)),
        }

    
    def run_experiment(
        self,
        base_path: str | Path,
        output_dir: str | Path,
        models_to_process: Optional[list] = None,
        max_samples_per_model: Optional[int] = None,
        baseline_threshold: float = 0.3,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Run spectrogram explainability experiment on dataset.
        
        :param base_path: Base path to dataset
        :param output_dir: Output directory
        :param models_to_process: List of model folders to process (None = all)
        :param max_samples_per_model: Max samples per model folder
        :param baseline_threshold: Minimum prediction threshold
        :param resume: Resume from checkpoint
        :return: DataFrame with results
        """
        
        base_path = Path(base_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saliency_dir = output_dir / "saliency_maps"
        saliency_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("🔬 Spectrogram Occlusion Explainability Experiment")
        print("=" * 70)
        print(f"📁 Dataset: {base_path}")
        print(f"📊 Output: {output_dir}")
        print(f"🗺️  Saliency maps: {saliency_dir}")
        print(f"🔧 Method: {self.method.upper()}")
        if self.method == 'rise':
            print(f"   RISE masks: {self.n_masks}")
        else:
            print(f"   Patch size: {self.patch_size}, Stride: {self.stride}")
        print(f"💾 Checkpoint: {'Enabled' if self.checkpoint else 'Disabled'}")
        print("=" * 70 + "\n")
        
        tmp_file = output_dir / "spectrogram_results_progress.csv"
        prev_results = []
        if os.path.exists(tmp_file):
            prev_results = pd.read_csv(tmp_file).to_dict('records')

        results = prev_results
        
        tmp_save_freq = 1
        tmp_file = output_dir / "spectrogram_results_progress.csv"
        
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
                        output_dir=saliency_dir,
                        baseline_threshold=baseline_threshold,
                        folder_name=folder.name
                    )
                    
                    if result:
                        results.append(result)

                        if len(results) % tmp_save_freq == 0:
                            pd.DataFrame(results).to_csv(tmp_file, index=False)
                            print(f"🔄 Auto-saved progress to {tmp_file}")
        
            if not results:
                print("\n⚠️  No results to save!")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            
            csv_path = output_dir / f"spectrogram_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            
            print("\n" + "=" * 70)
            print("✅ Experiment completed!")
            print("=" * 70)
            print(f"📊 Processed files: {len(df)}")
            print(f"📄 Results saved: {csv_path}")
            print(f"🗺️  Saliency maps: {saliency_dir}")
            print("=" * 70 + "\n")
            
            return df
    
        except Exception as e:
            print(f"\n\n❌ Critical error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            if results:
                pd.DataFrame(results).to_csv(tmp_file, index=False)
                print(f"⚠️  Crash! Progress auto-saved to {tmp_file}")
            raise


def visualize_aggregate_results(
    results_df: pd.DataFrame,
    output_dir: str | Path
):
    """
    Create aggregate visualizations from results.
    
    :param results_df: DataFrame with experiment results
    :param output_dir: Output directory for visualizations
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if results_df.empty:
        print("⚠️  Empty results, skipping visualizations")
        return
    
    print("\n📊 Generating aggregate visualizations...")
    
    # 1. Mean importance per model
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='folder', y='mean_importance', errorbar='sd')
    plt.title('Mean Importance per Model Generator', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Mean Importance (Δ Prediction)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_importance_per_model.png', dpi=300)
    plt.close()
    print("   ✅ mean_importance_per_model.png")
    
    # 2. Max importance per model
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='folder', y='max_importance', errorbar='sd')
    plt.title('Max Importance per Model Generator', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Max Importance (Δ Prediction)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'max_importance_per_model.png', dpi=300)
    plt.close()
    print("   ✅ max_importance_per_model.png")
    
    # 3. Distribution of importance
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=results_df, x='folder', y='mean_importance')
    plt.title('Distribution of Mean Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Mean Importance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'importance_distribution.png', dpi=300)
    plt.close()
    print("   ✅ importance_distribution.png")
    
    # 4. Correlation: baseline_pred vs mean_importance
    plt.figure(figsize=(10, 6))
    for folder in results_df['folder'].unique():
        folder_df = results_df[results_df['folder'] == folder]
        plt.scatter(folder_df['baseline_pred'], folder_df['mean_importance'], 
                   label=folder, alpha=0.6, s=50)
    plt.xlabel('Baseline Prediction (Fake Probability)', fontsize=12)
    plt.ylabel('Mean Importance', fontsize=12)
    plt.title('Baseline Prediction vs Mean Importance', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_vs_importance.png', dpi=300)
    plt.close()
    print("   ✅ prediction_vs_importance.png")
    
    # 5. Summary statistics table
    summary = results_df.groupby('folder').agg({
        'mean_importance': ['mean', 'std'],
        'max_importance': ['mean', 'std'],
        'baseline_pred': ['mean', 'std']
    }).round(4)
    
    summary.to_csv(output_dir / 'summary_statistics.csv')
    print("   ✅ summary_statistics.csv")
    
    print(f"\n✅ Aggregate visualizations saved to: {output_dir}\n")
