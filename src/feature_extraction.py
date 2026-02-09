from email.mime import audio
import numpy as np
import librosa

def extract_all_features(audio, sr, reference_audio=None):
    
    features = {}

    features['duration'] = len(audio) / sr

    f0, voiced_flag, voiced_probs = librosa.pyin(y=audio,
                                                 fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'))
    
    S = np.abs(librosa.stft(audio))
    S_db = librosa.power_to_db(S)

    rms_wave = librosa.feature.rms(y=audio)
    rms_spec = librosa.feature.rms(S=S)
    
    features['rms_wave'] = {
        "min": np.min(rms_wave),
        "mean": np.mean(rms_wave),
        "std": np.std(rms_wave),
        "max": np.max(rms_wave),
    }

    features['rms_spec'] = {
        "min": np.min(rms_spec),
        "mean": np.mean(rms_spec),
        "std": np.std(rms_spec),
        "max": np.max(rms_spec),
    }

    features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features['spectral_centroid'] = {
        'min': np.min(spectral_centroid),
        'mean': np.mean(spectral_centroid),
        'std': np.std(spectral_centroid),
        'max': np.max(spectral_centroid),
    }

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features['spectral_bandwidth'] = {
        'min': np.min(spectral_bandwidth),
        'mean': np.mean(spectral_bandwidth),
        'std': np.std(spectral_bandwidth),
        'max': np.max(spectral_bandwidth),
    }

    for roll_percent in [0.01, 0.85, 0.99]:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, 
                                                             roll_percent=roll_percent)
        features[f'spectral_rolloff_{int(roll_percent*100)}'] = {
            'min': np.min(spectral_rolloff),
            'mean': np.mean(spectral_rolloff),
            'std': np.std(spectral_rolloff),
            'max': np.max(spectral_rolloff),
        }

    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features['spectral_contrast'] = {
        'min': np.min(spectral_contrast),
        'mean': np.mean(spectral_contrast),
        'std': np.std(spectral_contrast),
        'max': np.max(spectral_contrast),
    }

    spectral_flatness = librosa.feature.spectral_flatness(y=audio)
    features['spectral_flatness'] = {
        'min': np.min(spectral_flatness),
        'mean':np.mean(spectral_flatness),
        'std':np.std(spectral_flatness),
        'max': np.max(spectral_flatness),
    }

    features['f0'] = {
        'min': np.nanmin(f0),
        'mean': np.nanmean(f0),
        'std': np.nanstd(f0),
        'max': np.nanmax(f0),
    }

    features['jitter'] = compute_jitter_extended(audio, sr, f0=f0)
    features['shimmer'] = compute_shimmer_extended(audio, sr)
    features['hnr'] = compute_hnr(audio, sr)
    features['gne'] = compute_gne(audio, sr)

    features['breath_count'] = detect_breaths(audio, sr)
    features['intonation_pattern'] = compute_intonation_pattern(audio, sr, f0=f0)
    features['voice_breaks'] = detect_voice_breaks(audio, sr)
    features['rhythm_stats'] = compute_rhythm_stats(audio, sr)

    return features

def compute_jitter(y, sr, f0=None):
    if f0 is None:
        f0 = librosa.pyin(y, fmin=80, fmax=1500, sr=sr)[0]
    f0 = f0[~np.isnan(f0)]

    if len(f0) < 2:
        return np.nan
    jitter = np.mean(np.abs(np.diff(f0))) / np.mean(f0)
    return jitter * 100

def compute_jitter_extended(y, sr, f0=None):
    """
    Oblicza różne typy jitter — bardziej rozbudowana wersja
    """
    if f0 is None:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                   fmax=librosa.note_to_hz('C7'), sr=sr)
    
    f0_valid = f0[~np.isnan(f0)]
    
    if len(f0_valid) < 2:
        return {
            'jitter_local': np.nan,
            'jitter_rap': np.nan,
            'jitter_ppq5': np.nan,
            'jitter_mean_absolute': np.nan,
            'jitter_std': np.nan,
            'jitter_range': np.nan,
        }
    
    periods = 1.0 / (f0_valid + 1e-8)
    
    jitter_local_abs = np.mean(np.abs(np.diff(periods)))
    jitter_local_pct = (jitter_local_abs / np.mean(periods)) * 100
    
    if len(periods) >= 3:
        jitter_rap_values = []
        for i in range(1, len(periods) - 1):
            avg_neighbor = np.mean([periods[i-1], periods[i], periods[i+1]])
            jitter_rap_values.append(abs(periods[i] - avg_neighbor))
        jitter_rap_pct = (np.mean(jitter_rap_values) / np.mean(periods)) * 100
    else:
        jitter_rap_pct = np.nan
    
    if len(periods) >= 5:
        jitter_ppq5_values = []
        for i in range(2, len(periods) - 2):
            avg_neighbor = np.mean([periods[i-2], periods[i-1], periods[i], 
                                    periods[i+1], periods[i+2]])
            jitter_ppq5_values.append(abs(periods[i] - avg_neighbor))
        jitter_ppq5_pct = (np.mean(jitter_ppq5_values) / np.mean(periods)) * 100
    else:
        jitter_ppq5_pct = np.nan
    
    jitter_mean_absolute_ms = jitter_local_abs * 1000
    
    jitter_std = np.std(np.abs(np.diff(periods))) / np.mean(periods) * 100
    
    jitter_range = (np.max(np.abs(np.diff(periods))) - np.min(np.abs(np.diff(periods)))) / np.mean(periods) * 100
    
    return {
        'jitter_local': jitter_local_pct,
        'jitter_rap': jitter_rap_pct,
        'jitter_ppq5': jitter_ppq5_pct,
        'jitter_mean_absolute_ms': jitter_mean_absolute_ms,
        'jitter_std': jitter_std,
        'jitter_range': jitter_range,
    }


def detect_breaths(y, sr, min_pause=0.2, energy_thresh=0.05):
    frame_length, hop_length = 2048, 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    breath_indices = np.where(rms < energy_thresh)[0]
    pauses = []

    if len(breath_indices) == 0: return 0
    curr_start = times[breath_indices[0]]
    for i in range(1, len(breath_indices)):
        if times[breath_indices[i]] - times[breath_indices[i-1]] > min_pause:
            pauses.append((curr_start, times[breath_indices[i-1]]))
            curr_start = times[breath_indices[i]]
    pauses.append((curr_start, times[breath_indices[-1]]))
    return len(pauses)

def compute_hnr(y, sr):
    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic
    hnr_db = 10 * np.log10(np.sum(harmonic ** 2) / (np.sum(noise ** 2) + 1e-8))
    return hnr_db

def compute_gne(y, sr, frame_length=512, hop_length=160):
    """
    Glottal-to-Noise Excitation Ratio
    """
    harm = librosa.effects.harmonic(y)
    S_harmonic = np.abs(librosa.stft(harm))
    
    S_full = np.abs(librosa.stft(y))
    S_noise = S_full - S_harmonic
    
    gne_db = 10 * np.log10(np.sum(S_harmonic**2 + 1e-8) / np.sum(S_noise**2 + 1e-8))
    return gne_db

def compute_shimmer(y, sr):
    frame_length = int(0.03 * sr)
    hop_length = int(0.015 * sr)
    amplitude_envelope = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    if len(amplitude_envelope) < 2:
        return np.nan
    
    shimmer = np.mean(np.abs(np.diff(amplitude_envelope))) / np.mean(amplitude_envelope)
    return shimmer * 100

def compute_shimmer_extended(y, sr):
    """
    Oblicza różne typy shimmer — bardziej rozbudowana wersja
    """
    frame_length = int(0.03 * sr)
    hop_length = int(0.015 * sr)
    amplitude_envelope = librosa.feature.rms(y=y, frame_length=frame_length, 
                                              hop_length=hop_length)[0]
    
    if len(amplitude_envelope) < 2:
        return {
            'shimmer_local': np.nan,
            'shimmer_apq3': np.nan,
            'shimmer_apq5': np.nan,
            'shimmer_dB': np.nan,
            'shimmer_std': np.nan,
            'shimmer_range': np.nan,
        }
    
    shimmer_local_abs = np.mean(np.abs(np.diff(amplitude_envelope)))
    shimmer_local_pct = (shimmer_local_abs / np.mean(amplitude_envelope)) * 100
    
    if len(amplitude_envelope) >= 3:
        shimmer_apq3_values = []
        for i in range(1, len(amplitude_envelope) - 1):
            avg_neighbor = np.mean([amplitude_envelope[i-1], amplitude_envelope[i], 
                                    amplitude_envelope[i+1]])
            shimmer_apq3_values.append(abs(amplitude_envelope[i] - avg_neighbor))
        shimmer_apq3_pct = (np.mean(shimmer_apq3_values) / np.mean(amplitude_envelope)) * 100
    else:
        shimmer_apq3_pct = np.nan
    
    if len(amplitude_envelope) >= 5:
        shimmer_apq5_values = []
        for i in range(2, len(amplitude_envelope) - 2):
            avg_neighbor = np.mean([amplitude_envelope[i-2], amplitude_envelope[i-1], 
                                    amplitude_envelope[i], amplitude_envelope[i+1], 
                                    amplitude_envelope[i+2]])
            shimmer_apq5_values.append(abs(amplitude_envelope[i] - avg_neighbor))
        shimmer_apq5_pct = (np.mean(shimmer_apq5_values) / np.mean(amplitude_envelope)) * 100
    else:
        shimmer_apq5_pct = np.nan
    
    amplitude_db = 20 * np.log10(amplitude_envelope + 1e-8)
    shimmer_db = np.mean(np.abs(np.diff(amplitude_db)))
    
    shimmer_std = np.std(np.abs(np.diff(amplitude_envelope))) / np.mean(amplitude_envelope) * 100
    
    shimmer_range = (np.max(np.abs(np.diff(amplitude_envelope))) - 
                     np.min(np.abs(np.diff(amplitude_envelope)))) / np.mean(amplitude_envelope) * 100
    
    return {
        'shimmer_local': shimmer_local_pct,
        'shimmer_apq3': shimmer_apq3_pct,
        'shimmer_apq5': shimmer_apq5_pct,
        'shimmer_dB': shimmer_db,
        'shimmer_std': shimmer_std,
        'shimmer_range': shimmer_range,
    }


def compute_intonation_pattern(y, sr, f0=None):
    # times = librosa.times_like(f0, sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    pitch_var = np.std(f0)
    # return {'f0_contour': f0.tolist(), 'pitch_variability': float(pitch_var), 'times': times.tolist()}
    return {'pitch_variability': float(pitch_var)}

def detect_voice_breaks(y, sr, threshold=0.1, min_duration_ms=50):
    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)

    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    breaks = energy < threshold

    counts = 0
    min_frames = int(min_duration_ms / (hop_length/sr*1000))
    count = 0
    for b in breaks:
        if b:
            count += 1
        else:
            if count >= min_frames:
                counts += 1
            count = 0
    if count >= min_frames:
        counts += 1

    return counts

def compute_rms_envelope(audio, sr=44100, frame_length=2048, hop_length=512):
    """
    Calculates RMS enveloper of the audio signal.
    :return: array with RMS for successive frames
    """
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms

def compute_rhythm_stats(audio, sr=44100):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    avg_onset_strength = np.mean(onset_env)
    max_onset_strength = np.max(onset_env)
    # beats_times = librosa.frames_to_time(beats, sr=sr)
    
    return {
        "tempo_bpm": tempo,
        "avg_onset_strength": avg_onset_strength,
        "max_onset_strength": max_onset_strength
        # "beats_times": beats_times.tolist()
    }