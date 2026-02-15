from __future__ import annotations
import sys
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml

import librosa

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_extraction import extract_all_features
from feature_calculate import append_update_features

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute physical features for FBP bands"
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "FBP_configs" / "fbp_band_features.yaml"),
        help="Path to the YAML configuration file",
    )
    return ap.parse_args()

def load_metadata_for_components(meta_path: Path, expected_components: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("bands", [])

def main():
    args = parse_args()
    config = load_yaml(Path(args.config))

    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    audio_cfg = config.get("audio", {})
    bands_cfg = config.get("fbp_bands", {})

    fbp_root = Path(dataset_cfg.get("fbp_result_path"))
    result_root = Path(output_cfg.get("result_path"))
    experiment_name = output_cfg.get("experiment_name", "fbp_bands")

    sr = int(audio_cfg.get("samplerate", 44100))
    components = bands_cfg.get("components", ["mixture"])
    components = set(components)

    # dictionary where to save extracted features from FBP bands
    output_root = result_root / experiment_name
    output_root.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Occlusion Band Features Extraction")
    print("=" * 70)
    print(f"Occlusion results: {fbp_root}")
    print(f"Output:            {output_root}")
    print(f"Sample rate:       {sr}")
    print(f"Components:        {', '.join(components)}")
    print("=" * 70)

    all_features = {}

    bands_root = fbp_root / "bands"
    if not bands_root.exists():
        print(f"[ERROR]: Bands directory not found: {bands_root}")
        return
    
    for model_dir in sorted(bands_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        track_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        track_iter = tqdm(track_dirs, desc=f"{model_name}", unit="track")

        for track_dir in track_iter:
            if not track_dir.is_dir():
                continue
            track_stem = track_dir.name

            for component in components:
                component_dir = track_dir / component
                if not component_dir.is_dir():
                    continue

                meta_path = component_dir / f"{track_stem}_bands_metadata.json"
                if not meta_path.exists():
                    print(f"[WARN] Missing meta json: {meta_path}")
                    continue

                bands = load_metadata_for_components(meta_path, component)
                
                if not bands:
                    continue
                    
                band_iter = tqdm(
                    bands,
                    desc=f"{model_name} - {track_stem} - {component}",
                    unit="band",
                    leave=False
                )    

                for band in band_iter:
                    component = band.get("component", "mixture")
                    low = band["low"]
                    high = band["high"]
                    importance = band["importance"]
                    abs_importance = band["abs_importance"]
                    ptype = band.get("type", "unknown")

                    wav_name = (
                        f"{track_stem}__{component}__{int(low)}-{int(high)}Hz_{ptype}_{importance:+.3f}.wav"
                    )
                    wav_path = component_dir / "freq_batches" / wav_name
                    if not wav_path.exists():
                        print(f"[WARN] Missing wav file: {wav_path}")
                        continue

                    y, _ = librosa.load(wav_path, sr=sr, mono=True)
                    feats = extract_all_features(y, sr)

                    feats_with_meta = dict(feats)
                    band_meta = {
                        "component": component,
                        "importance": importance,
                        "abs_importance": abs_importance,
                        "low_freq": low,
                        "high_freq": high,
                        "band_type": ptype,
                        "model": model_name,
                        "track_stem": track_stem,
                    }

                    model_dict = all_features.setdefault(model_name, {})
                    track_entry = model_dict.setdefault(
                        track_stem, 
                        {"type": "band", "bands": {}}
                    )

                    band_id = f"{component}_{low:.1f}_{high:.1f}Hz"
                    track_entry["bands"][band_id] = {
                        "features": feats_with_meta,
                        "band_meta": band_meta,
                    }
    features_path = output_root / "fbp_band_features.json"
    append_update_features(all_features, features_path)

    print("Saved fbp band features to:", features_path)

if __name__ == "__main__":
    main()