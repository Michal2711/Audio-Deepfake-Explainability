# scr/sonics_api.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, List
import numpy as np
import tempfile
import soundfile as sf
import os
import time
import random

from gradio_client import Client, handle_file
import httpx
from httpx import HTTPStatusError, WriteTimeout, ConnectTimeout
import socket

from sonics import HFAudioClassifier
import torch
import librosa

@dataclass
class RemoteSonnics:
    """
        Remote Sonnics predictor via Gradio API with retry logic
    """
    
    space: str
    model_time: int
    api_name: str = "/predict"
    model_type: str = "SpecTTTra-α"
    max_retries: int = 20
    initial_delay: float = 2.0
    max_delay: float = 60.0


    def __post_init__(self):        
        print(f"🔗 Connecting to space: {self.space}")
        print(f"🎯 Model type: {self.model_type}")
        print(f"⚙️  Max retries: {self.max_retries}")

        socket.setdefaulttimeout(180.0)
        self.client = Client(self.space)
        print("✅ Connected successfully")
        print(f"⏱️  Timeout set to: 180 seconds")


    def predict(self, audio_wave: np.ndarray, sr: int) -> float:
        """
        Predict with retry for network errors (502, 503, 504, WriteTimeout).
        Uses exponential backoff with jitter.

        :param audio_wave: Audio waveform
        :param sr: Sample rate
        :return: Fake probability (0-1)
        """
        
        tmp_path = None
        
        for attempt in range(self.max_retries):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                    sf.write(tmp_path, audio_wave, sr)
                                
                result = self.client.predict(
                    audio_file=handle_file(tmp_path),
                    model_type=self.model_type,
                    duration=f"{self.model_time}s",
                    api_name=self.api_name
                )
                
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        print(f"[Warning] Could not delete temp file {tmp_path}: {e}")
                
                fake_prob = next(
                    (item["confidence"] for item in result["confidences"] 
                     if item["label"] == "Fake"), 
                    0.0
                )
                return float(fake_prob)
            
            except (WriteTimeout, ConnectTimeout) as e:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_delay * (2 ** attempt) + random.uniform(0, 1), 
                        self.max_delay
                    )
                    print(f"[Warning] WriteTimeout - file too large (attempt {attempt + 1}/{self.max_retries})")
                    print(f"[Info] Retrying after {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"[Error] WriteTimeout - failed after {self.max_retries} attempts")
                    raise RuntimeError("WriteTimeout: Could not upload audio") from e
                    
            except HTTPStatusError as e:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as cleanup_err:
                        print(f"[Warning] Could not delete temp file after HTTP error: {cleanup_err}")
                
                if e.response.status_code in [502, 503, 504]:
                    if attempt < self.max_retries - 1:
                        delay = min(
                            self.initial_delay * (2 ** attempt) + random.uniform(0, 1), 
                            self.max_delay
                        )
                        print(f"[Warning] HTTP {e.response.status_code} error "
                              f"(attempt {attempt + 1}/{self.max_retries})")
                        print(f"[Info] Retrying after {delay:.2f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"[Error] Failed after {self.max_retries} attempts "
                              f"with HTTP {e.response.status_code}")
                        raise
                else:
                    print(f"[Error] HTTP {e.response.status_code} - not retrying")
                    raise
                    
            except Exception as e:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as cleanup_err:
                        print(f"[Warning] Could not delete temp file after error: {cleanup_err}")
                
                print(f"[Error] Unexpected error in predict: {type(e).__name__}: {e}")
                raise
        
        raise RuntimeError(f"Failed to get prediction after {self.max_retries} attempts")
    
    def predict_from_file(self, audio_path: Union[str, Path]) -> float:
        """
        Predict directly from file (uses API without conversion).
        BEST for remote - uses exactly the same pipeline as training.

        :param audio_path: Path to audio file
        :return: Fake probability (0-1)
        """
        
        audio_path = str(audio_path)
        
        for attempt in range(self.max_retries):
            try:
                
                result = self.client.predict(
                    audio_file=handle_file(audio_path),
                    model_type=self.model_type,
                    duration=f"{self.model_time}s",
                    api_name=self.api_name
                )
                
                fake_prob = next(
                    (item["confidence"] for item in result["confidences"] 
                     if item["label"] == "Fake"), 
                    0.0
                )
                print(f"      → Fake prob: {fake_prob:.4f}")
                return float(fake_prob)
            
            except (WriteTimeout, ConnectTimeout) as e:
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.initial_delay * (2 ** attempt) + random.uniform(0, 1), 
                        self.max_delay
                    )
                    print(f"[Warning] WriteTimeout - plik zbyt duży (attempt {attempt + 1}/{self.max_retries})")
                    print(f"[Info] Retrying after {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"[Error] WriteTimeout - failed after {self.max_retries} attempts")
                    raise RuntimeError(f"WriteTimeout: Could not upload {audio_path}") from e
                    
            except HTTPStatusError as e:
                if e.response.status_code in [502, 503, 504]:
                    if attempt < self.max_retries - 1:
                        delay = min(
                            self.initial_delay * (2 ** attempt) + random.uniform(0, 1), 
                            self.max_delay
                        )
                        print(f"[Warning] HTTP {e.response.status_code} error "
                              f"(attempt {attempt + 1}/{self.max_retries})")
                        print(f"[Info] Retrying after {delay:.2f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        raise
                else:
                    raise
            except Exception as e:
                print(f"[Error] Unexpected error: {type(e).__name__}: {e}")
                raise
        
        raise RuntimeError(f"Failed to get prediction after {self.max_retries} attempts")
    
    def predict_batch_from_files(self, audio_paths: List[Union[str, Path]], 
                                  verbose: bool = True,
                                  **kwargs) -> List[float]:
        """
        Predict on multiple files (batch processing).
        
        :param audio_paths: List of audio file paths
        :param verbose: Print progress
        :param kwargs: Ignored (for compatibility with LocalSonnics)
        :return: List of fake probabilities
        """
        probs = []
        for idx, path in enumerate(audio_paths):
            if verbose:
                print(f"   Predicting {idx+1}/{len(audio_paths)}: {Path(path).name}")
            prob = self.predict_from_file(path)
            probs.append(prob)
        return probs

@dataclass
class LocalSonnics:
    """Local predictor with GPU/CPU support"""
    
    model_name: str
    device: str = 'cuda'
    
    def __post_init__(self):
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'
        
        print(f"📥 Loading model: {self.model_name}")
        print(f"🔧 Target device: {self.device}")
        
        self.model = HFAudioClassifier.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    @classmethod
    def from_pretrained(cls, name: str, device: str = 'cuda'):
        return cls(model_name=name, device=device)
    
    def predict(self, audio_wave: np.ndarray, sr: int) -> float:
        """
        Predict fake probability for audio waveform.
        
        :param audio_wave: Audio waveform (mono)
        :param sr: Sample rate
        :return: Fake probability (0-1)
        """
        
        with torch.no_grad():
            t = torch.tensor(audio_wave).float().unsqueeze(0).to(self.device)
            raw = self.model(t)
            return torch.sigmoid(raw).cpu().item()
    
    def predict_from_file(self, audio_path: Union[str, Path], 
                          sr: int = 44100, 
                          duration: float = None) -> float:
        """
        Predict directly from file (uses librosa.load).
        Consistent with AudioDataset used in LIME.

        :param audio_path: Path to audio file
        :param sr: Target sample rate
        :param duration: Duration to load (seconds), None = all
        :return: Fake probability (0-1)
        """
        
        y, _ = librosa.load(str(audio_path), sr=sr, duration=duration, mono=True)
        return self.predict(y, sr)
    
    def predict_batch_from_files(self, audio_paths: List[Union[str, Path]], 
                                  sr: int = 44100,
                                  duration: float = None,
                                  verbose: bool = True,
                                  **kwargs) -> List[float]:
        """
        Predict on multiple files (batch processing).
        
        :param audio_paths: List of audio file paths
        :param sr: Target sample rate
        :param duration: Duration to load (seconds), None = all
        :param verbose: Print progress
        :param kwargs: Additional arguments (ignored)
        :return: List of fake probabilities
        """
        
        probs = []
        for idx, path in enumerate(audio_paths):
            if verbose:
                print(f"   Predicting {idx+1}/{len(audio_paths)}: {Path(path).name}")
            
            y, _ = librosa.load(str(path), sr=sr, duration=duration, mono=True)
            prob = self.predict(y, sr)
            probs.append(prob)
            
            if verbose:
                print(f"      → Fake prob: {prob:.4f}")
        
        return probs

def predict_from_file(predictor: Union[LocalSonnics, RemoteSonnics], 
                      audio_path: Union[str, Path],
                      **kwargs) -> float:
    """
    Unified prediction from file (works for both Local and Remote).
    
    :param predictor: LocalSonnics or RemoteSonnics instance
    :param audio_path: Path to audio file
    :param kwargs: Additional args (sr, duration for Local only)
    :return: Fake probability (0-1)
    """
    return predictor.predict_from_file(audio_path, **kwargs)

def predict_batch_from_files(predictor: Union[LocalSonnics, RemoteSonnics],
                              audio_paths: List[Union[str, Path]],
                              verbose: bool = True,
                              **kwargs) -> List[float]:
    """
    Unified batch prediction from files (works for both Local and Remote).
    
    :param predictor: LocalSonnics or RemoteSonnics instance
    :param audio_paths: List of audio file paths
    :param verbose: Print progress
    :param kwargs: Additional args (sr, duration for Local only, ignored for Remote)
    :return: List of fake probabilities
    """
    return predictor.predict_batch_from_files(audio_paths, verbose=verbose, **kwargs)