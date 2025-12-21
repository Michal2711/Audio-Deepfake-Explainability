# scripts/run_experiment.py
from __future__ import annotations

import os
import sys
import warnings

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

from dsp_band_ops import FBPConfig, FrequencyBandPerturbation
from sonics_api import LocalSonnics, RemoteSonnics


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_fbp_config(base_cfg: dict, bands_cfg: dict) -> FBPConfig:
    trans = bands_cfg.get("transition", {})
    cfg = FBPConfig(
        model_time=int(base_cfg.get("model_time", 120)),
        sr=int(base_cfg.get("sample_rate", 44100)),
        use_mel=bool(base_cfg.get("use_mel", False)),
        n_mels=int(base_cfg.get("n_mels", 128)),
        use_separation=bool(base_cfg.get("use_separation", False)),
        separation_model=str(base_cfg.get("separation_model", "spleeter:2stems")),
        separation_targets=tuple(base_cfg.get("separation_targets", ("vocals0", "accompaniment0"))),
        bands_preset=str(bands_cfg.get("preset", "default")),
        custom_bands=None,
        attenuation=float(bands_cfg.get("attenuation", 0.0)),
        transition_mode=str(trans.get("mode", "rel")),
        transition_hz=float(trans.get("hz", 200.0)),
        transition_rel=float(trans.get("rel", 0.2)),
        transition_min_hz=float(trans.get("min_hz", 20.0)),
        transition_max_hz=float(trans.get("max_hz", 2000.0)),
        n_fft=int(base_cfg.get("n_fft", 2048)),
        hop_length=int(base_cfg.get("hop_length", 512)),
        normalize_loudness=bool(base_cfg.get("normalize_loudness", True)),
    )
    presets = bands_cfg.get("presets", {})
    preset_name = bands_cfg.get("preset", "default")
    if preset_name in presets:
        cfg.custom_bands = [tuple(map(int, pair)) for pair in presets[preset_name]]
    return cfg


def build_predictor(exp_cfg: dict, base_cfg: dict):
    """Builds a predictor with optional retry configuration"""
    if bool(exp_cfg.get("local", False)):
        model_name = str(exp_cfg.get("local_model", "awsaf49/sonics-spectttra-alpha-120s"))
        predictor = LocalSonnics.from_pretrained(model_name)
    else:
        retry_cfg = exp_cfg.get("retry", {})
        
        predictor = RemoteSonnics(
            space=str(exp_cfg.get("remote_space", "awsaf49/sonics-fake-song-detection")),
            model_time=int(base_cfg.get("model_time", 120)),
            api_name=str(exp_cfg.get("remote_api_name", "/predict")),
            model_type=str(exp_cfg.get("remote_model_type", "SpecTTTra-α")),
            max_retries=int(retry_cfg.get("max_retries", 10)),
            initial_delay=float(retry_cfg.get("initial_delay", 3.0)),
            max_delay=float(retry_cfg.get("max_delay", 120.0)),
        )
    return predictor

def main():
    ap = argparse.ArgumentParser(description="Run Frequency Band Perturbation experiment")
    ap.add_argument("--base", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--bands", default=str(ROOT / "configs" / "bands.yaml"))
    ap.add_argument("--exp", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--resume", action="store_true", help="Resume experiment from checkpoint")
    ap.add_argument("--force-reprocess", action="store_true", help="Force reprocess all files from scratch")
    ap.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    ap.add_argument("--show-failed", action="store_true", help="Show only list of failed files and exit")
    args = ap.parse_args()

    base_cfg = load_yaml(Path(args.base))
    bands_cfg = load_yaml(Path(args.bands))
    exp_cfg = load_yaml(Path(args.exp))

    base_path = Path(exp_cfg.get("base_path", "../../Data/FakeRealMusicOriginal"))
    output_root = Path(exp_cfg.get("output_root", "results/FakeRealMusicOriginal"))
    experiment_name = str(exp_cfg.get("experiment_name", "exp"))
    limit_per_folder = exp_cfg.get("limit_per_folder", None)
    
    checkpoint_dir = None
    if not args.no_checkpoint:
        checkpoint_dir = output_root / experiment_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.show_failed:
        if checkpoint_dir is None or not (checkpoint_dir / "failed_files.json").exists():
            print("❌ No failed files data found. Run the experiment with checkpointing.")
            sys.exit(1)
        
        from dsp_band_ops import ExperimentCheckpoint
        checkpoint = ExperimentCheckpoint(checkpoint_dir)
        failed = checkpoint.get_failed_files()
        
        if not failed:
            print("✅ No failed files!")
        else:
            print(f"\n❌ Failed files ({len(failed)}):\n")
            for idx, fail_info in enumerate(failed, 1):
                print(f"{idx}. {Path(fail_info['file_path']).name}")
                print(f"   Error: {fail_info['error']}")
                print(f"   Time: {fail_info['timestamp']}\n")

        sys.exit(0)

    predictor = build_predictor(exp_cfg, base_cfg)
    fbp_cfg = build_fbp_config(base_cfg, bands_cfg)

    fbp = FrequencyBandPerturbation(
        predictor=predictor, 
        cfg=fbp_cfg,
        checkpoint_dir=checkpoint_dir
    )
    
    print("\n" + "=" * 70)
    print("🎵 Frequency Band Perturbation Experiment")
    print("=" * 70)
    print(f"📁 Data Source: {base_path}")
    print(f"📊 Output: {output_root / experiment_name}")
    print(f"🔧 Checkpoint: {'Enabled' if checkpoint_dir else 'Disabled'}")

    if checkpoint_dir:
        print(f"📂 Checkpoint dir: {checkpoint_dir}")
        if args.resume:
            print("🔄 Mode: Resume from last checkpoint")
        elif args.force_reprocess:
            print("🔄 Mode: Force reprocess all files")
        else:
            print("🔄 Mode: Normal (auto-resume if checkpoint exists)")

    if limit_per_folder:
        print(f"📏 Limit per folder: {limit_per_folder}")

    print("=" * 70 + "\n")

    previous_results = None
    if args.resume and checkpoint_dir and not args.force_reprocess:
        previous_results = fbp.load_previous_results(output_root, experiment_name)

    try:
        df = fbp.run_experiment(
            base_path=base_path, 
            limit_per_folder=limit_per_folder,
            resume=args.resume or (not args.force_reprocess),  # Default resume=True
            force_reprocess=args.force_reprocess,
            previous_results=previous_results
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user (Ctrl+C)")
        print("💾 Progress saved in checkpoint.")
        print(f"📂 Checkpoint: {checkpoint_dir}")
        print("\n💡 To resume, restart with --resume flag")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Critical error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        if checkpoint_dir:
            print(f"\n💾 Progress saved in: {checkpoint_dir}")
            print("💡 To resume, restart with --resume flag")

        sys.exit(1)

    if df.empty:
        print("\n⚠️  WARNING: No results to save!")
        print("All files may have failed or have already been processed.")

        if checkpoint_dir:
            from dsp_band_ops import ExperimentCheckpoint
            checkpoint = ExperimentCheckpoint(checkpoint_dir)
            stats = checkpoint.get_stats()
            failed = checkpoint.get_failed_files()

            print(f"\n📊 Checkpoint statistics:")
            print(f"   Processed: {stats['total_processed']}")
            print(f"   Failed: {stats['total_failed']}")

            if failed:
                print(f"\n❌ Examples of failed files (last 5):")
                for fail_info in failed[-5:]:
                    print(f"   - {Path(fail_info['file_path']).name}")
                    print(f"     {fail_info['error'][:100]}")
        
        sys.exit(1)

    extra_meta = {
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "librosa": librosa.__version__,
        },
        "config_files": {
            "base": str(Path(args.base)),
            "bands": str(Path(args.bands)),
            "experiment": str(Path(args.exp)),
        },
        "execution": {
            "resumed": args.resume,
            "force_reprocess": args.force_reprocess,
            "checkpoint_enabled": checkpoint_dir is not None,
        }
    }

    save = fbp.save_experiment_results(
        results_df=df,
        output_dir=output_root,
        experiment_name=experiment_name,
        extra_meta=extra_meta,
        is_merged=(previous_results is not None)
    )

    print("\n📊 Generating visualizations...")
    vis_root = Path(save["experiment_dir"]) / "visualizations"
    
    try:
        fbp.visualize_results(df, output_dir=vis_root / "results")
        print("   ✅ Result visualizations")
    except Exception as e:
        print(f"   ⚠️  Error visualizing results: {e}")

    try:
        fbp.visualize_embedding(df, output_dir=vis_root / "embeddings")
        print("   ✅ Embedding visualizations")
    except Exception as e:
        print(f"   ⚠️  Error visualizing embeddings: {e}")

    print("\n" + "=" * 70)
    print("✅ Experiment completed successfully!")
    print("=" * 70)
    print(f"📄 CSV  -> {save['results_csv']}")
    print(f"📋 JSON -> {save['params_json']}")
    print(f"📁 Dir  -> {save['experiment_dir']}")
    
    if checkpoint_dir:
        from dsp_band_ops import ExperimentCheckpoint
        checkpoint = ExperimentCheckpoint(checkpoint_dir)
        stats = checkpoint.get_stats()
        failed = checkpoint.get_failed_files()

        print(f"\n📊 Final statistics:")
        print(f"   Successfully processed: {stats['total_processed'] - stats['total_failed']}")
        print(f"   Failed: {stats['total_failed']}")
        print(f"   Total number of results in CSV: {len(df)}")

        if failed:
            print(f"\n⚠️  {len(failed)} files failed")
            print(f"   Run: python {Path(__file__).name} --show-failed")
            print(f"   to see details")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
