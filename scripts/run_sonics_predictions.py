# scripts/run_LIME_experiment.py
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('h5py').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

import argparse
from pathlib import Path
import yaml
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonic_predictions import (
    run_sonics_predictions
)

from sonics_api import LocalSonnics, RemoteSonnics

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_experiment_config(config: dict, output_dir: Path, experiment_name: str):
    """
    Save experiment configuration to output directory.
    
    :param config: Configuration dictionary
    :param output_dir: Output directory
    :param experiment_name: Experiment name
    :param num_samples: Number of LIME samples used
    """
    config_dir = output_dir / experiment_name / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_path = config_dir / f"config_{timestamp}.yaml"
    
    config_with_meta = {
        'experiment_info': {
            'name': experiment_name,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat(),
        },
        **config
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_with_meta, f, default_flow_style=False, allow_unicode=True)
    
    print(f"💾 Config saved: {config_path}")
    return config_path

def main():
    ap = argparse.ArgumentParser(description="Run SONICS predictions for fake song detection")
    ap.add_argument("--config", default=str(ROOT / "configs/AudioLIME_configs" / "lime_experiment.yaml"))
    args = ap.parse_args()

    config = load_yaml(Path(args.config))
    
    dataset_cfg = config.get('dataset', {})
    model_cfg = config.get('model', {})
    output_cfg = config.get('output', {})
    
    dataset_path = dataset_cfg.get('dataset_path')
    result_path = Path(output_cfg.get('result_path'))
    experiment_name = output_cfg.get('experiment_name', 'lime_exp')
    explanations_path = result_path / experiment_name / "full_track" / "predictions.json"
    full_track_output_dir = result_path / experiment_name / "full_track"

    config_path = save_experiment_config(
        config=config,
        output_dir=result_path,
        experiment_name=experiment_name,
    )
    
    print("\n" + "=" * 70)
    print("🔍 LIME Explainability Experiment")
    print("=" * 70)
    print(f"📁 Dataset: {dataset_path}")
    print(f"📊 Output: {result_path / experiment_name}")
    print(f"⚙️  Config: {config_path}")
    print("=" * 70 + "\n")
    
    is_local = model_cfg.get('local', False)
    
    if is_local:
        print("🔧 Setting up LOCAL model...")
        import torch
        
        model_name = model_cfg.get('local_model', 'awsaf49/sonics-spectttra-alpha-120s')
        device = model_cfg.get('device', 'cuda')
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
            device = 'cpu'
        
        predictor = LocalSonnics.from_pretrained(
            name=model_name,
            device=device
        )
        
        print(f"✅ Local model ready on {device}\n")
        
    else:
        print("🔧 Setting up REMOTE API...")
        
        retry_cfg = model_cfg.get('retry', {})
        
        predictor = RemoteSonnics(
            space=model_cfg.get('remote_space', 'awsaf49/sonics-fake-song-detection'),
            model_time=model_cfg.get('model_time', 120),
            model_type=model_cfg.get('remote_model_type', 'SpecTTTra-α'),
            max_retries=retry_cfg.get('max_retries', 10),
            initial_delay=retry_cfg.get('initial_delay', 3.0),
            max_delay=retry_cfg.get('max_delay', 120.0)
        )
        
        print("✅ Remote API ready\n")
    
    try:
        run_sonics_predictions(
            predictor=predictor,
            dataset_path=dataset_path,
            explanations_path=str(explanations_path),
            sample_rate = dataset_cfg.get('sample_rate', 44100),
            threshold=0.5
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Critical error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ Experiment completed successfully!")
    print("=" * 70)
    print(f"⚙️  Configuration: {config_path}")
    print(f"📄 Explanations: {str(explanations_path)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
