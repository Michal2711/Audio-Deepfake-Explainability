# scripts/run_FBP_experiment.py
from __future__ import annotations

import os
import sys
import warnings

from sonics.models import model

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['PYTHONWARNINGS'] = 'ignore'

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('h5py').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np
import librosa
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dsp_band_ops import FrequencyBandPerturbation
from sonics_api import LocalSonnics, RemoteSonnics


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_predictor(model_cfg: dict):
    """Builds a predictor with optional retry configuration"""
    if bool(model_cfg.get("local", False)):
        model_name = str(model_cfg.get("local_model", "awsaf49/sonics-spectttra-alpha-120s"))
        predictor = LocalSonnics.from_pretrained(model_name)
    else:
        retry_cfg = model_cfg.get("retry", {})
        
        predictor = RemoteSonnics(
            space=str(model_cfg.get("remote_space", "awsaf49/sonics-fake-song-detection")),
            model_time=int(model_cfg.get("model_time", 120)),
            api_name=str(model_cfg.get("remote_api_name", "/predict")),
            model_type=str(model_cfg.get("remote_model_type", "SpecTTTra-α")),
            max_retries=int(retry_cfg.get("max_retries", 10)),
            initial_delay=float(retry_cfg.get("initial_delay", 3.0)),
            max_delay=float(retry_cfg.get("max_delay", 120.0)),
        )
    return predictor

def save_experiment_config(config: dict, output_dir: Path, experiment_name: str):
    """
    Save experiment configuration to output directory.
    
    :param config: Configuration dictionary
    :param output_dir: Output directory
    :param experiment_name: Experiment name
    """
    import yaml
    from datetime import datetime
    
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = config_dir / f"config_{timestamp}.yaml"
    
    config_with_meta = {
        'experiment_info': {
            'name': experiment_name,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat()
        },
        **config
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_with_meta, f, default_flow_style=False, allow_unicode=True)
    
    print(f"💾 Config saved: {config_path}")
    return config_path

def main():
    ap = argparse.ArgumentParser(description="Run Frequency Band Perturbation experiment")
    ap.add_argument("--config", default=str(ROOT / "configs" / "fbp_experiment.yaml"))
    ap.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    ap.add_argument("--resume", action="store_true", help="Resume experiment from checkpoint")
    args = ap.parse_args()

    config = load_yaml(Path(args.config))

    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    bands_cfg = config.get("bands", {})
    spectrogram_cfg = config.get("spectrogram", {})
    explain_cfg = config.get("explainability", {})
    output_cfg = config.get("output", {})
    checkpoint_cfg = config.get("checkpoint", {})

    base_path = Path(dataset_cfg.get("base_path"))
    output_root = Path(output_cfg.get("result_path"))
    experiment_name = str(output_cfg.get("experiment_name", "exp"))
    
    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_experiment_config(config, output_dir, experiment_name)

    checkpoint_dir = None
    if checkpoint_cfg.get('enabled', True) and not args.no_checkpoint:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(model_cfg)

    fbp = FrequencyBandPerturbation(
        predictor=predictor,
        preset = bands_cfg.get("preset", "default"),
        presets = bands_cfg.get("presets", {}),
        attenuation = float(bands_cfg.get("attenuation", 0.0)),
        transition_mode = str(bands_cfg.get("transition", {}).get("mode", "rel")),
        transition_hz = float(bands_cfg.get("transition", {}).get("hz", 200.0)),
        transition_rel = float(bands_cfg.get("transition", {}).get("rel", 0.2)),
        transition_min_hz = float(bands_cfg.get("transition", {}).get("min_hz", 20.0)),
        transition_max_hz = float(bands_cfg.get("transition", {}).get("max_hz", 2000.0)),
        sr = int(spectrogram_cfg.get("sample_rate", 44100)),
        duration=int(spectrogram_cfg.get("model_time", 120)),
        use_mel = bool(spectrogram_cfg.get("use_mel", False)),
        n_mels = int(spectrogram_cfg.get("n_mels", 128)),
        n_fft = int(spectrogram_cfg.get("n_fft", 2048)),
        hop_length = int(spectrogram_cfg.get("hop_length", 512)),
        win_length=int(spectrogram_cfg.get("win_length", 2048)),
        n_iter=int(spectrogram_cfg.get("n_iter", 32)),
        use_separation=bool(explain_cfg.get("use_separation", False)),
        separation_model=str(explain_cfg.get("separation_model", "spleeter:2stems")),
        separation_targets=tuple(explain_cfg.get("separation_targets", ("vocals0", "accompaniment0"))),
        normalize_loudness=bool(explain_cfg.get("normalize_loudness", True)),
        lufs=float(explain_cfg.get("lufs", -14.0)), 
        checkpoint_dir=checkpoint_dir
    )

    previous_results = None
    if args.resume and checkpoint_dir and not args.force_reprocess:
        previous_results = fbp.load_previous_results(output_root, experiment_name)

    try:
        df = fbp.run_experiment(
            base_path=base_path,
            output_dir=output_dir,
            models_to_process=dataset_cfg.get('models_to_process'),
            max_samples_per_model=dataset_cfg.get('max_samples_per_model'),
            results_path=output_dir / f"fbp_results.json"
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted (Ctrl+C)")
        if checkpoint_dir:
            print(f"💾 Progress saved in: {checkpoint_dir}")
            print("💡 Resume with --resume flag")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Critical error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if df.empty:
        print("\n⚠️  No results generated!")
        sys.exit(1)

    print("\n📊 Generating visualizations...")
    viz_dir = output_dir / "aggregate_visualizations"
    
    try:
        fbp.visualize_results(df, output_dir=viz_dir)
        print("   ✅ Result visualizations")
    except Exception as e:
        print(f"   ⚠️  Error visualizing results: {e}")

    print("\n" + "=" * 70)
    print("🎉 All done!")
    print("=" * 70)
    print(f"💾 Configuration: {config_path}")
    print(f"📈 Aggregate visualizations: {viz_dir}")
    print(f"📄 Results CSV: {list(output_dir.glob('fbp_results_*.csv'))[-1]}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
