from __future__ import annotations
import re
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
        description="Compute physical features for AudioLIME components"
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "AudioLIME_configs" / "lime_comp_features.yaml"),
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
    components_cfg = config.get("lime_comp_features", {})

    lime_root = Path(dataset_cfg.get("lime_result_path"))
    result_root = Path(output_cfg.get("result_path"))
    experiment_name = output_cfg.get("experiment_name", "lime_comp_features")

    sr = int(audio_cfg.get("samplerate", 44100))
    components = components_cfg.get("components", ["mixture"])
    components = set(components)

    output_root = result_root / experiment_name
    output_root.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("AudioLIME Component Features Extraction")
    print("=" * 70)
    print(f"AudioLIME results: {lime_root}")
    print(f"Output:            {output_root}")
    print(f"Sample rate:       {sr}")
    print(f"Components:        {', '.join(components)}")
    print("=" * 70)

    all_features = {}

    full_root = lime_root / "full_track"
    if not full_root.exists():
        print(f"[ERROR]: Full track directory not found: {full_root}")
        return
    
    explanations_path = full_root / "explanations.json"
    with open(explanations_path, "r", encoding="utf-8") as f:
        expl_df = json.load(f)

    for model_dir in sorted(full_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        track_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        track_iter = tqdm(track_dirs, desc=f"{model_name}", unit="track")

        for track_dir in track_iter:
            if not track_dir.is_dir():
                continue
            track_stem = Path(track_dir.name).stem
            safe_track_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', track_stem)

            components_dir = track_dir / "separated_components"
            if not components_dir.exists():
                print(f"[WARN] Components directory not found: {components_dir}")
                continue

            # iterating over components wav files like bass0.wav etc.
            for component_file in components_dir.glob("*.wav"):
                component_name = component_file.stem
                if component_name not in components:
                    continue

                meta = expl_df.get(model_name, {}).get(safe_track_name, {}).get("explanations", {})
                importance = meta.get("component_influences", {}).get(component_name, 0.0)

                y, _ = librosa.load(component_file, sr=sr, mono=True)
                feats = extract_all_features(y, sr)

                feats_with_meta = dict(feats)
                feats_with_meta["importance"] = importance

                model_dict = all_features.setdefault(model_name, {})
                track_entry = model_dict.setdefault(
                    safe_track_name, 
                    {"type": "full_track", "components": {}}
                )

                track_entry["components"][component_name] = {
                    "features": feats_with_meta,
                    "component_meta": {
                        "importance": importance,
                        "abs_importance": abs(importance),
                        "component_type": "POSITIVE" if importance >= 0 else "NEGATIVE",
                        "model": model_name,
                        "track_stem": track_stem,
                        "component_name": component_name,
                    }
                }

            
    features_path = output_root / "audiolime_component_features.json"
    append_update_features(all_features, features_path)

    print("Saved AudioLIME component features to:", features_path)

if __name__ == "__main__":
    main()