# src/feature_visualizations.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import librosa.display
import json

from feature_extraction import compute_rms_envelope, compute_rhythm_stats

def plot_waveform(audio, sr=44100, title="Waveform", output_path=None):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()

def plot_rms_envelope(audio, sr=44100, output_path=None, title="RMS Envelope"):
    """
    Rysuje wykres enveloper RMS w czasie.
    """
    times, rms = compute_rms_envelope(audio, sr)
    plt.figure(figsize=(10, 4))
    plt.plot(times, rms, label="RMS Envelope")
    plt.xlabel("Time [s]")
    plt.ylabel("RMS")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()

def plot_rhythm_statistics(audio, sr=44100, output_dir=None, prefix=""):
    """
    Wizualizacja rytmu oraz impulsów.
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    rhythm_stats = compute_rhythm_stats(audio, sr)
    
    plt.figure(figsize=(10, 4))
    times, rms = compute_rms_envelope(audio, sr)
    plt.plot(times, rms, label='RMS Envelope')
    plt.vlines(rhythm_stats['beats_times'], ymin=0, ymax=np.max(rms), color='r', alpha=0.75, label='Beats')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude / RMS')
    plt.title(f"Rhythm and Beats ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_rhythm_beats.png")
    plt.close()

    return rhythm_stats

def plot_enhanced_visualizations(audio, sr=44100, prefix="", output_dir=None):
    """
    Generate enhanced visualizations including waveform
    and classical feature spectrograms.
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    plot_waveform(audio, sr=sr, title=f"Waveform ({prefix})",
                  output_path=output_dir / f"{prefix}_waveform.png")

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title(f'Chroma Features ({prefix})')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_chroma.png")
    plt.close()

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC ({prefix})')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_mfcc.png")
    plt.close()

    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec_contrast, x_axis='time')
    plt.colorbar()
    plt.title(f'Spectral Contrast ({prefix})')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_spectral_contrast.png")
    plt.close()

def plot_spectral_summary(audio, sr=44100, output_dir=None, prefix=""):
    """
    Creates a comprehensive visualization: RMS, spectrogram with centroid/rolloff, spectral contrast.
    """
    import matplotlib.pyplot as plt
    import librosa.display
    import numpy as np
    from pathlib import Path

    rms = librosa.feature.rms(y=audio)[0]
    times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    mean_rms = np.mean(rms)

    S = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
    times_spec = librosa.frames_to_time(np.arange(len(centroid)), sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    plt.figure(figsize=(13, 12))

    # 1. RMS
    plt.subplot(3, 1, 1)
    plt.semilogy(times_rms, rms, label='RMS Energy')
    plt.axhline(mean_rms, color="red", ls="--", label=f"mean(RMS): {mean_rms:.3f}")
    plt.legend()
    plt.title("RMS Energy")

    # 2. Spectrogram + spectral features
    plt.subplot(3, 1, 2)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="log")
    plt.plot(times_spec, centroid, color="w", lw=1.3, label='Spectral centroid')
    plt.plot(times_spec, rolloff, color="c", lw=1.5, label='Spectral rolloff (0.85)')
    plt.legend(loc="upper right")
    plt.title("log Power spectrogram")
    plt.colorbar(img, format='%+2.0f dB')

    # 3. Spectral contrast
    plt.subplot(3, 1, 3)
    librosa.display.specshow(contrast, x_axis='time')
    plt.title("Spectral contrast")
    plt.ylabel("Frequency bands")
    plt.colorbar()
    plt.xlabel("Time [s]")

    plt.tight_layout()
    if output_dir:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / f"{prefix}_spectral_summary.png")
    plt.close()

def plot_all_waveforms(original_audio, components, component_names, sr, output_path, prefix=""):
    n_comps = len(components)
    fig, axs = plt.subplots(n_comps+1, 1, figsize=(12, 2.5*(n_comps+1)))
    # Original audio
    axs[0].plot(original_audio, color='grey')
    axs[0].set_title("Original Audio - Waveform")
    axs[0].set_xlim(0, len(original_audio))

    # Components
    for i, (audio, name) in enumerate(zip(components, component_names)):
        axs[i+1].plot(audio)
        axs[i+1].set_title(f"{name.capitalize()} - Waveform")
        axs[i+1].set_xlim(0, len(audio))
    plt.tight_layout(pad=2.0)
    outfile = Path(output_path) / f"{prefix}_all_waveforms.png"
    plt.savefig(outfile)
    plt.close()

def plot_all_spectrograms(original_audio, components, component_names, sr, output_path, prefix=""):
    import librosa
    import librosa.display
    n_comps = len(components)
    fig, axs = plt.subplots(n_comps, 2, figsize=(13, 3*n_comps))
    for i, (audio, name) in enumerate(zip(components, component_names)):
        # Original audio
        S = librosa.feature.melspectrogram(y=original_audio, sr=sr)
        S_dB = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=axs[i,0])
        axs[i,0].set_title("Original Audio")

        # Component
        S_c = librosa.feature.melspectrogram(y=audio, sr=sr)
        S_c_dB = librosa.amplitude_to_db(S_c, ref=np.max)
        librosa.display.specshow(S_c_dB, x_axis='time', y_axis='mel', sr=sr, ax=axs[i,1])
        axs[i,1].set_title(f'{name.capitalize()}')

    for row in range(n_comps):
        for col in range(2):
            axs[row, col].label_outer()
    plt.tight_layout(pad=2.0)
    outfile = Path(output_path) / f"{prefix}_all_spectrograms.png"
    plt.savefig(outfile)
    plt.close()

def plot_f0_contour(y: np.ndarray, sr: int, f0: np.ndarray, times: np.ndarray, title: str = "Fundamental Frequency (pYIN)", output_dir: Path = None, prefix: str = ""):
    """
    Plot the power spectrogram (dB) with f0 contour.

    :param y: audio signal (1D numpy array)
    :param sr: sampling frequency
    :param f0: array with f0 values (fundamental frequency)
    :param times: array with times corresponding to f0 values
    :param title: plot title
    """
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=2)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_spectral_contrast.png")
    plt.close()

def plot_mel_spectrogram_with_f0(y: np.ndarray, sr: int, f0: np.ndarray, times: np.ndarray, title: str = "Mel Spectrogram (Vocal) with f0", output_dir: Path = None, prefix: str = ""):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    
    # f0 contour (times and frequencies)
    ax.plot(times, librosa.hz_to_mel(f0), label='f0', color='cyan', linewidth=2)
    ax.legend(loc='upper right')
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(output_dir / f"{prefix}_mel_f0.png")
    plt.close()