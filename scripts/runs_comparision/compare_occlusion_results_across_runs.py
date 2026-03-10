# compare_occlusion_results_across_runs.py

import json
import argparse
from pathlib import Path
from typing import Sequence, Dict, Any, Tuple
import re
import yaml

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def setup_professional_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 16

    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5

    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.major.width"] = 1.5
    plt.rcParams["ytick.major.width"] = 1.5

    sns.set_palette("husl")


PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#ff9896",
    "#98df8a",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
]


def try_num(s: str) -> int:
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="ignore")
    match = re.match(r"^(\d+)", s)
    return int(match.group(1)) if match else 999999


def extract_run_label(file_path: str) -> str:
    path = Path(file_path)
    name = str(path).lower()

    if "minus14" not in name and "minus23" not in name:
        return "Original"
    elif "base" in name and "minus14" in name:
        return "m14_base"
    elif "base" in name and "minus23" in name:
        return "m23_base"
    elif "mp3_192" in name and "minus14" in name:
        return "m14_mp3_192"
    elif "mp3_192" in name and "minus23" in name:
        return "m23_mp3_192"
    elif "noise_snr30" in name and "minus14" in name:
        return "m14_noise_snr30"
    elif "noise_snr30" in name and "minus23" in name:
        return "m23_noise_snr30"
    elif "resample22k" in name and "minus14" in name:
        return "m14_resample_22k"
    elif "resample22k" in name and "minus23" in name:
        return "m23_resample22k"
    elif "reverb_room" in name and "minus14" in name:
        return "m14_reverb_room"
    elif "reverb_room" in name and "minus23" in name:
        return "m23_reverb_room"
    else:
        return path.parent.name if path.parent.name != "." else path.stem[:20]

def get_freq_unit(label: str) -> str:
    """Dynamiczna jednostka: 'Mel' jeśli 'mel' w nazwie, 'Hz' dla STFT/innych."""
    name_lower = label.lower()
    if 'mel' in name_lower:
        return 'Mel'
    elif 'stft' in name_lower:
        return 'Hz'
    else:
        return 'Mel'

def load_single_occlusion_root(occ_root: Path, run_label: str) -> pd.DataFrame:
    rows = []

    sal_root = occ_root / "saliency_maps"
    if not sal_root.exists():
        print(f"[ERROR] saliency_maps directory not found: {sal_root}")
        return pd.DataFrame()

    unit = get_freq_unit(run_label)

    for model_dir in sorted(sal_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        track_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        for track_dir in track_dirs:
            track_stem = track_dir.name
            all_dir = track_dir / "top_windows" / "all"
            if not all_dir.exists():
                continue

            json_files = sorted(all_dir.glob("*.json"))
            if not json_files:
                continue

            for jf in json_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load {jf}: {e}")
                    continue

                file_name = meta.get("file_name", track_stem)
                windows = meta.get("windows", [])
                if not windows:
                    continue

                for w in windows:
                    rank = int(w.get("rank", 0))

                    t_start = int(w.get("t_start", 0))
                    t_end = int(w.get("t_end", 0))
                    f_start = float(w.get("f_start", 0.0))
                    f_end = float(w.get("f_end", 0.0))

                    start_sec = float(w.get("start_time_sec", 0.0))
                    end_sec = float(w.get("end_time_sec", 0.0))
                    importance = float(w.get("importance", np.nan))
                    abs_importance = float(w.get("abs_importance", np.nan))
                    wtype = w.get("type", "unknown")

                    time_label = f"{start_sec:.1f}-{end_sec:.1f}s"
                    freq_label = f"{int(f_start)}-{int(f_end)}{unit}"
                    window_label = f"{time_label}, {freq_label}"

                    rows.append({
                        "data_source": model_name,
                        "track_stem": track_stem,
                        "track_index": try_num(track_stem),
                        "file_name": file_name,
                        "rank": rank,
                        "t_start": t_start,
                        "t_end": t_end,
                        "f_start": f_start,
                        "f_end": f_end,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "window_label": window_label,
                        "importance": importance,
                        "abs_importance": abs_importance,
                        "window_type": wtype,
                        "run": run_label,
                    })

    df = pd.DataFrame(rows)
    return df



def load_occlusion_windows_for_runs(
    occ_roots: Sequence[str], max_rank: int | None = None
) -> Tuple[pd.DataFrame, str]:
    dfs = []
    runs_labels = ""

    for p in occ_roots:
        occ_root = Path(p)
        run_label = extract_run_label(p)
        runs_labels += f"{run_label}_"

        print(f"📂 Loading Occlusion windows from: {occ_root} (run: {run_label})")
        df_run = load_single_occlusion_root(occ_root, run_label)
        if df_run.empty:
            print(f"[WARN] No data loaded from {occ_root}")
            continue

        if max_rank is not None:
            df_run = df_run[df_run["rank"] <= max_rank].copy()

        dfs.append(df_run)
        print(f"✅ Loaded {len(df_run)} window rows from {occ_root}")

    if not dfs:
        raise ValueError("No Occlusion data loaded from any run!")

    df_all = pd.concat(dfs, ignore_index=True)

    key_cols = [
        "data_source",
        "track_stem",
        "t_start",
        "t_end",
        "f_start",
        "f_end",
    ]
    keys_per_run = [df.groupby(key_cols).size() > 0 for _, df in df_all.groupby("run")]

    common_mask = keys_per_run[0].reindex(keys_per_run[0].index).fillna(False)
    for mask in keys_per_run[1:]:
        common_mask &= mask.reindex(common_mask.index).fillna(False)

    idx = df_all.set_index(key_cols).index
    common_idx = common_mask[common_mask].index
    df_common = df_all[idx.isin(common_idx)].copy()

    df_common = df_common.sort_values(
        ["data_source", "start_sec", "f_start", "run", "track_index"]
    ).reset_index(drop=True)

    print(
        f"✅ Common Occlusion data: {len(df_common)} rows across "
        f"{df_common['data_source'].nunique()} sources"
    )
    print(f"   Runs: {sorted(df_common['run'].unique())}")

    return df_common, runs_labels.strip("_")

def plot_occlusion_windows_importances(
    df_common: pd.DataFrame, output_dir: Path | None = None
):
    setup_professional_style()
    sns.set_theme(style="whitegrid")

    run_names = sorted(df_common["run"].unique())
    legend_runs = " vs ".join(run_names)

    providers = sorted(df_common["data_source"].unique())

    for prov in providers:
        dprov = df_common[df_common["data_source"] == prov].copy()
        if dprov.empty:
            continue

        tracks = sorted(dprov["track_stem"].unique(), key=try_num)
        idx_pos = {t: i for i, t in enumerate(tracks)}
        dprov["file_index"] = dprov["track_stem"].map(idx_pos)

        windows_order = (
            dprov[["window_label", "start_sec", "f_start"]]
            .drop_duplicates()
            .sort_values(["start_sec", "f_start"])["window_label"]
            .tolist()
        )
        if not windows_order:
            continue

        g = sns.FacetGrid(
            dprov,
            col="window_label",
            col_order=windows_order,
            hue="run",
            height=3.0,
            aspect=1.3,
            col_wrap=5,
            sharey=False,
            palette="husl",
            legend_out=False,
        )

        g.map_dataframe(
            sns.lineplot,
            x="file_index",
            y="importance",
            legend=False,
            linewidth=1.5,
            alpha=0.8,
        )

        g.map_dataframe(
            sns.scatterplot,
            x="file_index",
            y="importance",
            legend=False,
            s=80,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )

        g.set_axis_labels("file index", "importance")
        g.set_titles(col_template="{col_name}")
        unit = get_freq_unit(df_common['run'].iloc[0])
        g.fig.suptitle(
            f"{prov}: Occlusion window importance vs file index ({unit}) ({legend_runs})",
            y=1.02,
            fontsize=12,
        )

        handles, labels = g.axes.flat[0].get_legend_handles_labels()
        g.fig.legend(
            handles,
            labels,
            title="Run",
            loc="upper left",
            bbox_to_anchor=(0.82, 0.85),
            frameon=True,
            fancybox=True,
            shadow=False,
            fontsize=10,
        )

        short_labels = [t[:18] + "..." if len(t) > 18 else t for t in tracks]
        index_text = "\n".join(f"{i:2d}: {lab}" for i, lab in enumerate(short_labels))
        g.fig.text(
            0.82,
            0.45,
            f"File Mapping:\n{index_text}",
            fontsize=8.8,
            va="top",
            ha="left",
            bbox=dict(
                facecolor="white",
                edgecolor="#d1d5db",
                boxstyle="round,pad=0.4",
                alpha=0.95,
            ),
        )

        g.fig.subplots_adjust(right=0.78)
        plt.subplots_adjust(bottom=0.06)

        if output_dir:
            outfile = output_dir / f"{prov}_occlusion_windows_by_track.png"
            plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"💾 Saved: {outfile}")

        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Occlusion window importance – runs comparison (FacetGrid style)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml (with list of Occlusion roots in 'files')",
    )
    args = parser.parse_args()

    with open(Path(args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    occ_roots = config.get("files", [])
    if not occ_roots:
        print("❌ No Occlusion roots specified in config['files']!")
        return

    max_rank = config.get("max_rank", None)
    if max_rank is not None:
        print(f"Limiting windows to rank <= {max_rank}")

    df_common, runs_labels = load_occlusion_windows_for_runs(occ_roots, max_rank)

    output_cfg = config.get("output", {})
    output_dir = (
        Path(output_cfg.get("result_path", "results/Occlusion/Runs_comparison"))
        / runs_labels
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_occlusion_windows_importances(df_common, output_dir)
    print(f"✅ All Occlusion plots saved to {output_dir}")


if __name__ == "__main__":
    main()
