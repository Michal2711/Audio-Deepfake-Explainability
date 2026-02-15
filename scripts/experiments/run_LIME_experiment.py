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

from lime_explainer import (
    run_lime_experiment_safe
)
from lime_visualizations import (
    visualize_explanations,
    visualize_explanations_by_model
)
from sonics_api import LocalSonnics, RemoteSonnics
from lime_explainer import load_existing_explanations

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_experiment_config(config: dict, output_dir: Path, experiment_name: str, num_samples: int):
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
            'num_samples_lime': num_samples
        },
        **config
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_with_meta, f, default_flow_style=False, allow_unicode=True)
    
    print(f"💾 Config saved: {config_path}")
    return config_path

def main():
    ap = argparse.ArgumentParser(description="Run LIME experiment for fake song detection")
    ap.add_argument("--config", default=str(ROOT / "configs/AudioLIME_configs" / "lime_experiment.yaml"))
    ap.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = ap.parse_args()

    config = load_yaml(Path(args.config))
    
    dataset_cfg = config.get('dataset', {})
    model_cfg = config.get('model', {})
    lime_cfg = config.get('lime', {})
    output_cfg = config.get('output', {})
    viz_cfg = config.get('visualization', {})
    explanations_variants_cfg = config.get('explanation_variants', {})
    
    dataset_path = dataset_cfg.get('dataset_path')
    result_path = Path(output_cfg.get('result_path'))
    experiment_name = output_cfg.get('experiment_name', 'lime_exp')
    num_samples_lime = lime_cfg.get('num_samples_lime', 50)
    full_track_explanations = explanations_variants_cfg.get('full_track_explanations', True)
    segmented_explanations = explanations_variants_cfg.get('segmented_explanations', False)
    segment_duration = explanations_variants_cfg.get('segment_duration', 10)
    explanations_path = result_path / experiment_name / "full_track" / "explanations.json"
    segmented_explanations_path = result_path / experiment_name / "segmented" / "segmented_explanations.json"
    full_track_output_dir = result_path / experiment_name / "full_track"
    segmented_output_dir = result_path / experiment_name / "segmented"

    config_path = save_experiment_config(
        config=config,
        output_dir=result_path,
        experiment_name=experiment_name,
        num_samples=num_samples_lime
    )

    checkpoint_dir = None
    if not args.no_checkpoint:
        checkpoint_dir = result_path / experiment_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🔍 LIME Explainability Experiment")
    print("=" * 70)
    print(f"📁 Dataset: {dataset_path}")
    print(f"📊 Output: {result_path / experiment_name}")
    print(f"⚙️  Config: {config_path}")
    print(f"🎯 Models to explain: {', '.join(lime_cfg.get('models_to_explain', []))}")
    print(f"📏 Samples per model: {lime_cfg.get('max_samples_explain')}")
    print(f"🔬 LIME samples: {num_samples_lime}")
    print(f"💾 Checkpoint: {'Enabled' if checkpoint_dir else 'Disabled'}")
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
        df, explanations = run_lime_experiment_safe(
            predictor=predictor,
            model_time=model_cfg.get('model_time', 120),
            explain=lime_cfg.get('explain', True),
            max_samples_explain=lime_cfg.get('max_samples_explain', 5),
            dataset_path=dataset_path,
            num_samples_lime=num_samples_lime,
            models_to_explain=lime_cfg.get('models_to_explain', []),
            ids_to_explain=lime_cfg.get('ids_to_explain', list(range(10))),
            checkpoint_dir=checkpoint_dir,
            explanations_path=str(explanations_path),
            features_output_dir_full=str(full_track_output_dir),
            features_output_dir_segmented = str(segmented_output_dir),
            full_track_explanations=full_track_explanations,
            segmented_explanations=segmented_explanations,
            segment_duration=segment_duration,
            segmented_explanations_path=str(segmented_explanations_path)
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
    
    print("\n📊 Generating visualizations...")

    if viz_cfg.get('overall', True):
        viz_path_overall = result_path / experiment_name / "overall_visualizations"
        try:
            visualize_explanations(explanations, output_dir=str(viz_path_overall))
            print(f"✅ Overall visualizations: {viz_path_overall}")
        except Exception as e:
            print(f"⚠️  Error in overall visualizations: {e}")
    
    if viz_cfg.get('per_model', True):
        viz_path_per_model = result_path / experiment_name / "visualizations_per_model"
        try:
            visualize_explanations_by_model(explanations, output_dir=str(viz_path_per_model))
            print(f"✅ Per-model visualizations: {viz_path_per_model}")
        except Exception as e:
            print(f"⚠️  Error in per-model visualizations: {e}")

    print("\n" + "=" * 70)
    print("✅ Experiment completed successfully!")
    print("=" * 70)
    print(f"⚙️  Configuration: {config_path}")
    print(f"📄 Explanations: {str(explanations_path)}")
    if viz_cfg.get('overall'):
        print(f"📊 Overall viz: {viz_path_overall}")
    if viz_cfg.get('per_model'):
        print(f"📊 Per-model viz: {viz_path_per_model}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
