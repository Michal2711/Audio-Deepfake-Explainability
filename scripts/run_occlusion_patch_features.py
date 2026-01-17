from __future__ import annotations
import os
import sys
import json
from pathlib import Path
import argparse
import yaml

import librosa

ROOT = Path(__file__).resolve().parents[1]
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
        description="Compute physical features for occlusion patches"
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "occlusion_patch_features.yaml"),
        help="Path to the YAML configuration file",
    )
    return ap.parse_args()


def load_metadata_for_group(meta_path: Path, expected_group: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("windows", [])


def main():
    args = parse_args()
    config = load_yaml(Path(args.config))

    dataset_cfg = config.get("dataset", {})
    output_cfg = config.get("output", {})
    audio_cfg = config.get("audio", {})
    patches_cfg = config.get("occlusion_patches", {})

    occlusion_root = Path(dataset_cfg.get("occlusion_result_path"))
    result_root = Path(output_cfg.get("result_path"))
    experiment_name = output_cfg.get("experiment_name", "occlusion_patches")

    sr = int(audio_cfg.get("samplerate", 44100))
    groups = patches_cfg.get("groups", ["best", "most_influential"])
    groups = set(groups)

    # dictionary where to save extracted features from occlusion patches
    output_root = result_root / experiment_name
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Occlusion Patch Features Extraction")
    print("=" * 70)
    print(f"Occlusion results: {occlusion_root}")
    print(f"Output:            {output_root}")
    print(f"Sample rate:       {sr}")
    print(f"Groups:            {', '.join(groups)}")
    print("=" * 70)

    all_features = {}

    # structure: occlusion_root / saliency_maps / <modelname> / <trackstem> / topwindows / <group>/*.wav
    saliency_root = occlusion_root / "saliency_maps"
    if not saliency_root.exists():
        print(f"[ERROR] saliency_maps dir not found: {saliency_root}")
        return

    for model_dir in sorted(saliency_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        for track_dir in sorted(model_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            track_stem = track_dir.name
            windows_dir = track_dir / "top_windows"
            if not windows_dir.exists():
                continue

            for group in groups:
                group_dir = windows_dir / group
                if not group_dir.exists():
                    continue

                meta_path = group_dir / f"{track_stem}__{group}_occlusion_patches_from_list.json"
                if not meta_path.exists():
                    print(f"[WARN] Missing meta json: {meta_path}")
                    continue

                windows = load_metadata_for_group(meta_path, group)
                if not windows:
                    continue

                for win in windows:
                    rank = win["rank"]
                    importance = win["importance"]
                    abs_importance = win["abs_importance"]
                    tstart = win["t_start"]
                    tend = win["t_end"]
                    fstart = win["f_start"]
                    fend = win["f_end"]
                    start_time_sec = win["start_time_sec"]
                    end_time_sec = win["end_time_sec"]
                    ptype = win["type"]

                    wav_name = (
                        f"{track_stem}__{group}{rank}_patch_"
                        f"{ptype}_{abs_importance:.3f}_"
                        f"t{tstart}-{tend}_f{fstart}-{fend}.wav"
                    )
                    wav_path = group_dir / wav_name
                    if not wav_path.exists():
                        print(f"[WARN] Missing patch wav: {wav_path}")
                        continue

                    y, _ = librosa.load(wav_path, sr=sr, mono=True)
                    feats = extract_all_features(y, sr)

                    feats_with_meta = dict(feats)
                    occlusion_meta = {
                        "group": group,
                        "rank": int(rank),
                        "importance": float(importance),
                        "abs_importance": float(abs_importance),
                        "tstart": int(tstart),
                        "tend": int(tend),
                        "fstart": int(fstart),
                        "fend": int(fend),
                        "start_time_sec": float(start_time_sec),
                        "end_time_sec": float(end_time_sec),
                        "patch_type": ptype,
                        "model": model_name,
                        "track_stem": track_stem,
                    }

                    model_dict = all_features.setdefault(model_name, {})
                    track_entry = model_dict.setdefault(
                        track_stem,
                        {"type": "patch", "patches": {}},
                    )
                    patch_id = f"{group}_rank{rank}"
                    track_entry["patches"][patch_id] = {
                        "features": feats_with_meta,
                        "occlusion_meta": occlusion_meta,
                    }

    features_path = output_root / "occlusion_patches_features.json"
    append_update_features(all_features, features_path)

    print("Saved occlusion patch features to:", features_path)


if __name__ == "__main__":
    main()
