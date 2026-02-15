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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_calculate import run_features_extraction

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_experiment_config(config: dict, output_dir: Path, experiment_name: str):
    """
    Save experiment configuration to output directory.
    
    :param config: Configuration dictionary
    :param output_dir: Output directory
    :param experiment_name: Experiment name
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
    ap = argparse.ArgumentParser(description="Extract features for audio dataset")
    ap.add_argument("--config", default=str(ROOT / "configs/Features_extraction" / "features_configs.yaml"))
    args = ap.parse_args()

    config = load_yaml(Path(args.config))
    
    dataset_cfg = config.get('dataset', {})
    output_cfg = config.get('output', {})
    feat_cfg = config.get('features', {})
    audio_cfg = config.get('audio', {})
    #     
    dataset_path = dataset_cfg.get('dataset_path')
    result_path = Path(output_cfg.get('result_path'))
    experiment_name = output_cfg.get('experiment_name', 'lime_exp')
    sample_rate = audio_cfg.get('sample_rate')

    full_track_output_dir = result_path / experiment_name / "full_track"
    segmented_output_dir = result_path / experiment_name / "segmented"

    max_samples = feat_cfg.get('max_samples', None)
    models_to_get_features = feat_cfg.get('models_to_get_features', ['dummy_model'])
    ids_to_get_features = feat_cfg.get('ids_to_get_features', [])
    full_track_features = feat_cfg.get('extract_full_track_features', True)
    segmented_features = feat_cfg.get('extract_segmented_features', False)
    segment_duration = feat_cfg.get('segment_duration', 10.0)

    config_path = save_experiment_config(
        config=config,
        output_dir=result_path,
        experiment_name=experiment_name,
    )

    print("\n" + "="*70)
    print("🚀 Starting feature extraction")
    print("="*70 + "\n")
    print(f"📁 Dataset: {dataset_path}")
    print(f"📊 Output: {result_path / experiment_name}")
    print(f"⚙️  Config: {config_path}")
    print(f"🎯 Models to get features: {', '.join(feat_cfg.get('models_to_get_features', []))}")
    print(f"📏 Samples per model: {feat_cfg.get('max_samples', None)}")
    print("=" * 70 + "\n")
    
    try:
        run_features_extraction(
            dataset_path=dataset_path,
            model_time=120.0,
            max_samples = max_samples,
            models_to_get_features=models_to_get_features,
            ids_to_get_features=ids_to_get_features,
            features_output_dir_full=full_track_output_dir,
            features_output_dir_segmented=segmented_output_dir,
            full_track_features = full_track_features,
            segmented_features = segmented_features,
            segment_duration=segment_duration,
            sample_rate=sample_rate,
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Critical error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
