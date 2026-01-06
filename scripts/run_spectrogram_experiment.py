# scripts/run_spectrogram_experiment.py
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('h5py').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectrogram_explainability import (
    SpectrogramExplainability,
    visualize_aggregate_results
)
from sonics_api import RemoteSonnics


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
    ap = argparse.ArgumentParser(description="Run Spectrogram Occlusion Explainability experiment")
    ap.add_argument("--config", default=str(ROOT / "configs" / "spectrogram_experiment.yaml"))
    ap.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    ap.add_argument("--visualize-only", action="store_true", help="Only generate aggregate visualizations")
    args = ap.parse_args()

    config = load_yaml(Path(args.config))
    
    dataset_cfg = config.get('dataset', {})
    model_cfg = config.get('model', {})
    spectrogram_cfg = config.get('spectrogram', {})
    occlusion_cfg = config.get('occlusion', {})
    output_cfg = config.get('output', {})
    checkpoint_cfg = config.get('checkpoint', {})
    
    base_path = Path(dataset_cfg.get('base_path'))
    output_root = Path(output_cfg.get('result_path'))
    experiment_name = output_cfg.get('experiment_name', 'spectrogram_exp')
    
    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = save_experiment_config(config, output_dir, experiment_name)

    checkpoint_dir = None
    if checkpoint_cfg.get('enabled', True) and not args.no_checkpoint:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if args.visualize_only:
        print("\n📊 Generating aggregate visualizations only...")
        csv_files = sorted(output_dir.glob("spectrogram_results_*.csv"))
        if not csv_files:
            print("❌ No results CSV found!")
            sys.exit(1)
        
        import pandas as pd
        latest_csv = csv_files[-1]
        print(f"📥 Loading: {latest_csv}")
        df = pd.read_csv(latest_csv)
        
        viz_dir = output_dir / "aggregate_visualizations"
        visualize_aggregate_results(df, viz_dir)
        
        print(f"✅ Visualizations saved to: {viz_dir}")
        sys.exit(0)
    
    is_local = model_cfg.get('local', False)
        
    if is_local:
        print("\n🔧 Setting up LOCAL model...")
        from sonics_api import LocalSonnics
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
        print("\n🔧 Setting up REMOTE API...")
        from sonics_api import RemoteSonnics
        
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
    
    explainability_cfg = config.get('explainability', {})
    method = explainability_cfg.get('method', 'rise')
    visualization_cfg = explainability_cfg.get('visualization', {})
    
    if method == 'rise':
        rise_cfg = explainability_cfg.get('rise', {})
        explainer = SpectrogramExplainability(
            predictor=predictor,
            sr=spectrogram_cfg.get('sr', 44100),
            duration=spectrogram_cfg.get('duration', 120),
            n_fft=spectrogram_cfg.get('n_fft', 2048),
            hop_length=spectrogram_cfg.get('hop_length', 512),
            win_length=spectrogram_cfg.get('win_length', 2048),
            n_mels=spectrogram_cfg.get('n_mels', 256),
            n_iter=spectrogram_cfg.get('n_iter', 256),
            method='rise',
            use_original_audio=False,
            n_masks=rise_cfg.get('n_masks', 500),
            mask_probability=rise_cfg.get('mask_probability', 0.5),
            checkpoint_dir=checkpoint_dir,
            highlight_percent=visualization_cfg.get('highlight_percent', 20.0),
            abs_threshold=visualization_cfg.get('abs_threshold', None)
        )
    else:  # occlusion
        occlusion_cfg = explainability_cfg.get('occlusion', {})
        explainer = SpectrogramExplainability(
            predictor=predictor,
            sr=spectrogram_cfg.get('sr', 44100),
            duration=spectrogram_cfg.get('duration', 120),
            n_fft=spectrogram_cfg.get('n_fft', 2048),
            hop_length=spectrogram_cfg.get('hop_length', 512),
            win_length=spectrogram_cfg.get('win_length', 2048),
            n_mels=spectrogram_cfg.get('n_mels', 128),
            n_iter=spectrogram_cfg.get('n_iter', 256),
            top_n_windows=occlusion_cfg.get('top_n_windows', 5),
            method='occlusion',
            use_original_audio=occlusion_cfg.get('use_original_audio', True),
            patch_size=tuple(occlusion_cfg.get('patch_size', [32, 32])),
            stride=tuple(occlusion_cfg.get('stride', [16, 16])),
            checkpoint_dir=checkpoint_dir,
            highlight_percent=visualization_cfg.get('highlight_percent', 20.0),
            abs_threshold=visualization_cfg.get('abs_threshold', None)
        )
    
    try:

        baseline_threshold = explainability_cfg.get('baseline_threshold', 0.3)

        df = explainer.run_experiment(
            base_path=base_path,
            output_dir=output_dir,
            models_to_process=dataset_cfg.get('models_to_process'),
            max_samples_per_model=dataset_cfg.get('max_samples_per_model'),
            baseline_threshold=baseline_threshold,
            resume=args.resume or (not args.no_checkpoint),
            results_path=output_dir / f"spectrogram_results_{method}.json"
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
    
    print("\n📊 Generating aggregate visualizations...")
    viz_dir = output_dir / "aggregate_visualizations"
    visualize_aggregate_results(df, viz_dir)
    
    print("\n" + "=" * 70)
    print("🎉 All done!")
    print("=" * 70)
    print(f"💾 Configuration: {config_path}")
    print(f"🗺️  Saliency maps: {output_dir / 'saliency_maps'}")
    print(f"📈 Aggregate visualizations: {viz_dir}")
    print(f"📄 Results CSV: {list(output_dir.glob('spectrogram_results_*.csv'))[-1]}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
