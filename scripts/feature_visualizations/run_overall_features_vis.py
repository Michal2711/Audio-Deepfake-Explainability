import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def load_yaml(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

PROFESSIONAL_COLORS = {
    "REAL": "#1f77b4",
    "ElevenLabs": "#ff7f0e",
    "SUNO": "#2ca02c",
    "SUNO_PRO": "#d62728",
    "UDIO": "#9467bd",
}

BOX_FILL_COLORS = {
    "REAL": "#aec7e8",
    "ElevenLabs": "#ffbb78",
    "SUNO": "#98df8a",
    "SUNO_PRO": "#ff9896",
    "UDIO": "#c5b0d5",
}

CORRECTNESS_COLORS = {
    "correct": "#2ca02c",
    "incorrect": "#d62728",
}

OUTCOME_COLORS = {
    "TP": "#2ca02c",
    "TN": "#1f77b4",
    "FP": "#ff7f0e",
    "FN": "#d62728",
}


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

def flatten_feature(feat_dict, prefix=''):
    result = {}

    for key, val in feat_dict.items():
        col_name = f"{prefix}_{key}" if prefix else key

        if isinstance(val, dict):
            stats_keys = {"min", "mean", "std", "max"}
            if stats_keys.intersection(val.keys()):
                for stat_name, stat_val in val.items():
                    result[f"{col_name}_{stat_name}"] = (
                        float(stat_val) if isinstance(stat_val, (int, float)) else np.nan
                    )
            else:
                nested = flatten_feature(val, prefix=col_name)
                result.update(nested)

        elif isinstance(val, list):
            if len(val) > 0 and all(isinstance(x, (int, float)) for x in val):
                result[f"{col_name}_mean"] = float(np.mean(val))
                result[f"{col_name}_min"] = float(np.min(val))
                result[f"{col_name}_max"] = float(np.max(val))
                result[f"{col_name}_std"] = float(np.std(val)) if len(val) > 1 else 0.0
            else:
                pass

        elif isinstance(val, (int, float)):
            result[col_name] = float(val)
        elif isinstance(val, bool):
            result[col_name] = val
        elif isinstance(val, str):
            result[col_name] = val

    return result

def load_fulltrack_features(json_path: Path):
    """
    Oczekuje struktury jak w features_full_track-2.json:
    {
      "ElevenLabs": {
        "track_key": {
          "type": "full_track",
          "segments": {
            "segment_id": "full_track",
            "features": { "mix": {...} },
            "segment_meta": {
              "model": "ElevenLabs",
              "track_stem": "...",
              ...
            }
          }
        },
        ...
      },
      "REAL": {...},
      "SUNO": {...},
      ...
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    type_mapping = {
        "ElevenLabs": "GENERATED",
        "REAL": "REAL",
        "SUNO": "GENERATED",
        "SUNO_PRO": "GENERATED",
        "UDIO": "GENERATED",
    }

    for model_name, tracks_dict in data.items():
        for track_key, track_data in tracks_dict.items():
            if not isinstance(track_data, dict) or "segments" not in track_data:
                continue

            segments = track_data.get("segments", {})
            features = segments.get("features", {})
            mix = features.get("mix", {})
            segment_meta = segments.get("segment_meta", {})

            track_stem = segment_meta.get("track_stem", track_key)

            row = {
                "model": model_name,
                "track_id": track_key,
                "track_stem": track_stem,
                "data_type": type_mapping.get(model_name, model_name),
            }

            flattened_mix = flatten_feature(mix)
            row.update(flattened_mix)
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("⚠️ Warning: no features loaded from full-track JSON!")
        return df, []

    exclude_cols = {"model", "track_id", "track_stem", "data_type"}
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    print("\n" + "=" * 80)
    print("✅ Full-track features loaded")
    print(f" • Models: {sorted(df['model'].unique().tolist())}")
    print(f" • Total tracks: {len(df)}")
    print(f" • Total numeric features: {len(feature_cols)}")
    print(f" • Sample features: {feature_cols[:10]}")
    print("=" * 80 + "\n")

    return df, feature_cols

def load_predictions(json_path: Path):
    """
    Oczekuje struktury jak w predictions.json:
    {
      "ElevenLabs": {
        "track_key": {
          "file_path": "...",
          "model": "ElevenLabs",
          "track_stem": "...",
          "prediction": 0.98,
          "predicted_class": "Fake" | "Real",
          "track_source": "Fake" | "Real"
        },
        ...
      },
      "REAL": {...},
      ...
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for model_name, tracks_dict in data.items():
        for track_key, info in tracks_dict.items():
            if not isinstance(info, dict):
                continue

            track_stem = info.get("track_stem", track_key)
            pred_score = info.get("prediction", np.nan)
            pred_class = info.get("predicted_class", None)
            true_source = info.get("track_source", None)

            true_label = true_source  # Real/Fake
            pred_label = pred_class

            if true_label is None or pred_label is None:
                outcome = "unknown"
                is_correct = False
            else:
                is_correct = (true_label == pred_label)
                if true_label == "Fake" and pred_label == "Fake":
                    outcome = "TP"
                elif true_label == "Fake" and pred_label == "Real":
                    outcome = "FN"
                elif true_label == "Real" and pred_label == "Fake":
                    outcome = "FP"
                else:
                    outcome = "TN"

            rows.append(
                {
                    "model": model_name,
                    "track_id": track_key,
                    "track_stem": track_stem,
                    "prediction_score": float(pred_score),
                    "pred_label": pred_label,
                    "true_label": true_label,
                    "is_correct": bool(is_correct),
                    "outcome": outcome,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        print("⚠️ Warning: no predictions loaded from JSON!")
        return df

    print("\n" + "=" * 80)
    print("✅ Predictions loaded")
    print(f" • Models: {sorted(df['model'].unique().tolist())}")
    print(f" • Total predictions: {len(df)}")
    print(
        f" • Outcomes counts:\n{df['outcome'].value_counts(dropna=False).to_string()}"
    )
    print("=" * 80 + "\n")

    return df

def merge_features_and_predictions(features_df, preds_df):
    merged = pd.merge(
        features_df,
        preds_df,
        on=["model", "track_stem"],
        how="inner",
        suffixes=("", "_pred"),
    )

    print("\n" + "=" * 80)
    print("✅ Merged features + predictions")
    print(f" • Merged rows: {len(merged)}")
    print(" • Columns sample:", merged.columns[:15].tolist())
    print("=" * 80 + "\n")

    return merged

def build_feature_groups(df, extra_exclude=None):
    """
    Grupuje cechy wg rdzenia, np.
      rms_wave_min, rms_wave_mean, rms_wave_std, rms_wave_max
      -> grupa 'rms_wave' + list[(col, 'min'/'mean'/...)]
    """
    if extra_exclude is None:
        extra_exclude = set()

    base_exclude = {
        "model",
        "track_id",
        "track_stem",
        "data_type",
        "prediction_score",
        "pred_label",
        "true_label",
        "is_correct",
        "outcome",
    }

    exclude_cols = base_exclude.union(extra_exclude)

    all_cols = [
        c
        for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() > 0
    ]

    feature_groups = defaultdict(list)
    for col in all_cols:
        parts = col.split("_")
        if len(parts) > 1 and parts[-1] in ["min", "mean", "std", "max"]:
            base_name = "_".join(parts[:-1])
            stat = parts[-1]
        else:
            base_name = col
            stat = "single"
        feature_groups[base_name].append((col, stat))

    return feature_groups

def format_influence_statistics_box(labels, plot_data):
    """
    labels: np. ['ElevenLabs\ncorrect', 'ElevenLabs\nincorrect', ...]
    plot_data: lista 1D arrayów (dane do boxplota)
    """
    rows = []
    header = ["Group", "Mean", "Std", "Count"]
    rows.append(header)

    for label, data in zip(labels, plot_data):
        if data is None or len(data) == 0:
            continue

        group_name = label.replace("\n", " ")

        mean_str = f"{np.mean(data):.4f}"
        std_str = f"{np.std(data):.4f}"
        count_str = f"{len(data)}"

        row = [group_name, mean_str, std_str, count_str]
        rows.append(row)

    if len(rows) == 1:
        return ""

    n_cols = len(rows[0])
    col_widths = [
        max(len(str(row[c])) for row in rows)
        for c in range(n_cols)
    ]

    def fmt_row(row):
        cells = []
        for i, (val, w) in enumerate(zip(row, col_widths)):
            text = str(val)
            if i == 0:
                cells.append(text.ljust(w))
            else:
                cells.append(text.rjust(w))
        return " ".join(cells)

    lines = []
    lines.append(fmt_row(rows[0]))
    lines.append("─" * (sum(col_widths) + 2 * (n_cols - 1)))

    for row in rows[1:]:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def add_bottom_stats_panel(fig, anchor_ax, text, width_frac=0.38, y_margin=0.04):
    if not text:
        return None

    bbox = anchor_ax.get_position()

    panel_width = bbox.width * width_frac
    left = bbox.x0 + (bbox.width - panel_width) / 2.0

    height = 0.10
    bottom = y_margin

    stats_ax = fig.add_axes([left, bottom, panel_width, height])
    stats_ax.axis("off")

    stats_ax.text(
        0.0, 1.0,
        text,
        ha="left", va="top",
        fontsize=9,
        transform=stats_ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.95,
            edgecolor="black",
            linewidth=1.0,
        ),
        family="monospace",
    )

    return stats_ax

def viz_features_by_model_and_global(df, output_root: Path):
    setup_professional_style()

    out_dir = output_root / "boxplots_by_model_global"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_groups = build_feature_groups(df)
    models = sorted(df["model"].dropna().unique())
    data_types = ["REAL", "GENERATED"]

    print("\n" + "=" * 80)
    print("Creating boxplots: features per model + REAL/GENERATED ...")
    print(f"Found {len(feature_groups)} feature groups\n")

    for feature_base, columns_list in sorted(feature_groups.items()):
        feature_folder = out_dir / feature_base
        feature_folder.mkdir(parents=True, exist_ok=True)

        stat_order = ["min", "mean", "std", "max"]
        columns_sorted = sorted(
            columns_list,
            key=lambda x: next(
                (i for i, stat in enumerate(stat_order) if stat == x[1]), 999
            ),
        )

        for col, stat in columns_sorted:
            stat_label = stat.upper() if stat != "single" else col

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            ax_models, ax_global = axes

            plot_data = []
            x_labels = []

            for model in models:
                data = df.loc[df["model"] == model, col].dropna()
                if len(data) > 0:
                    plot_data.append(data.values)
                    x_labels.append(model)

            if len(plot_data) == 0:
                plt.close(fig)
                continue

            bp = ax_models.boxplot(
                plot_data,
                tick_labels=x_labels,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                vert=True,
                whis=1.5,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=7),
                medianprops=dict(color="darkblue", linewidth=2),
                boxprops=dict(linewidth=1.5, color="black"),
                whiskerprops=dict(linewidth=1.5, color="black"),
                capprops=dict(linewidth=1.5, color="black"),
            )

            for i, patch in enumerate(bp["boxes"]):
                model = x_labels[i]
                color = BOX_FILL_COLORS.get(model, "#cccccc")
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor("black")
                patch.set_linewidth(2)

            for i, data in enumerate(plot_data):
                x = np.random.normal(i + 1, 0.05, size=len(data))
                ax_models.scatter(
                    x, data, alpha=0.35, s=25, c="black", ec="gray", linewidth=0.5
                )

            ax_models.set_ylabel("Value", fontsize=13, fontweight="bold")
            ax_models.set_title(
                f"{feature_base} – {stat_label}\nper model",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            ax_models.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
            ax_models.spines["top"].set_visible(False)
            ax_models.spines["right"].set_visible(False)
            for tick in ax_models.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")

            global_plot_data = []
            global_labels = []

            for dt in data_types:
                data = df.loc[df["data_type"] == dt, col].dropna()
                if len(data) > 0:
                    global_plot_data.append(data.values)
                    global_labels.append(dt)

            if global_plot_data:
                bp2 = ax_global.boxplot(
                    global_plot_data,
                    tick_labels=global_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=7),
                    medianprops=dict(color="darkblue", linewidth=2),
                    boxprops=dict(linewidth=1.5, color="black"),
                    whiskerprops=dict(linewidth=1.5, color="black"),
                    capprops=dict(linewidth=1.5, color="black"),
                )

                for i, patch in enumerate(bp2["boxes"]):
                    label = global_labels[i]
                    if label == "REAL":
                        color = "#1f77b4"
                    else:
                        color = "#7f7f7f"
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(2)

                for i, data in enumerate(global_plot_data):
                    x = np.random.normal(i + 1, 0.05, size=len(data))
                    ax_global.scatter(
                        x, data, alpha=0.35, s=25, c="black", ec="gray", linewidth=0.5
                    )

                ax_global.set_ylabel("Value", fontsize=13, fontweight="bold")
                ax_global.set_title(
                    f"{feature_base} – {stat_label}\nREAL vs GENERATED (all models)",
                    fontsize=13,
                    fontweight="bold",
                    pad=15,
                )
                ax_global.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
                ax_global.spines["top"].set_visible(False)
                ax_global.spines["right"].set_visible(False)
            else:
                ax_global.text(
                    0.5,
                    0.5,
                    "No data for REAL / GENERATED",
                    transform=ax_global.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="red",
                )
                ax_global.axis("off")

            fig.suptitle(
                f"Full-track feature analysis: {feature_base.replace('_', ' ').title()} – {stat_label}",
                fontsize=16,
                fontweight="bold",
                y=0.97,
            )
            plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

            stats_text = format_influence_statistics_box(x_labels, plot_data)
            add_bottom_stats_panel(fig, ax_models, stats_text, width_frac=0.45, y_margin=0.04)

            if global_plot_data:
                global_stats_text = format_influence_statistics_box(global_labels, global_plot_data)
                add_bottom_stats_panel(fig, ax_global, global_stats_text, width_frac=0.30, y_margin=0.04)

            out_file = feature_folder / f"{feature_base}_{stat_label}_by_model_global.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    print("=" * 80)
    print(f"✅ Saved by-model/global boxplots to: {out_dir}")
    print("=" * 80 + "\n")

def viz_features_correct_vs_incorrect(df, output_root: Path):
    setup_professional_style()

    out_dir = output_root / "boxplots_correct_incorrect"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_groups = build_feature_groups(df)
    models = sorted(df["model"].dropna().unique())
    correctness_levels = ["correct", "incorrect"]

    print("\n" + "=" * 80)
    print("Creating boxplots: features for correctly vs incorrectly classified ...")
    print(f"Found {len(feature_groups)} feature groups\n")

    for feature_base, columns_list in sorted(feature_groups.items()):
        feature_folder = out_dir / feature_base
        feature_folder.mkdir(parents=True, exist_ok=True)

        stat_order = ["min", "mean", "std", "max"]
        columns_sorted = sorted(
            columns_list,
            key=lambda x: next(
                (i for i, stat in enumerate(stat_order) if stat == x[1]), 999
            ),
        )

        for col, stat in columns_sorted:
            stat_label = stat.upper() if stat != "single" else col

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            ax_models, ax_global = axes

            plot_data = []
            x_labels = []

            for model in models:
                for corr_flag in correctness_levels:
                    mask = (df["model"] == model) & (
                        df["is_correct"] == (corr_flag == "correct")
                    )
                    data = df.loc[mask, col].dropna()
                    if len(data) > 0:
                        plot_data.append(data.values)
                        x_labels.append(f"{model}\n{corr_flag}")
            if not plot_data:
                plt.close(fig)
                continue

            bp = ax_models.boxplot(
                plot_data,
                tick_labels=x_labels,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                vert=True,
                whis=1.5,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=7),
                medianprops=dict(color="darkblue", linewidth=2),
                boxprops=dict(linewidth=1.5, color="black"),
                whiskerprops=dict(linewidth=1.5, color="black"),
                capprops=dict(linewidth=1.5, color="black"),
            )

            for i, patch in enumerate(bp["boxes"]):
                label = x_labels[i]
                if "\n" in label:
                    _, flag = label.split("\n", 1)
                    flag = flag.strip()
                else:
                    flag = label.strip()

                if flag == "correct":
                    color = CORRECTNESS_COLORS["correct"]
                elif flag == "incorrect":
                    color = CORRECTNESS_COLORS["incorrect"]
                else:
                    color = "#cccccc"

                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor("black")
                patch.set_linewidth(2)

            for i, data in enumerate(plot_data):
                x = np.random.normal(i + 1, 0.05, size=len(data))
                ax_models.scatter(
                    x, data, alpha=0.35, s=25, c="black", ec="gray", linewidth=0.5
                )

            ax_models.set_ylabel("Value", fontsize=13, fontweight="bold")
            ax_models.set_title(
                f"{feature_base} – {stat_label}\nper model (correct vs incorrect)",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            ax_models.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
            ax_models.spines["top"].set_visible(False)
            ax_models.spines["right"].set_visible(False)
            for tick in ax_models.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")

            global_plot_data = []
            global_labels = []
            for corr_flag in correctness_levels:
                mask = df["is_correct"] == (corr_flag == "correct")
                data = df.loc[mask, col].dropna()
                if len(data) > 0:
                    global_plot_data.append(data.values)
                    global_labels.append(corr_flag)

            if global_plot_data:
                bp2 = ax_global.boxplot(
                    global_plot_data,
                    tick_labels=global_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=7),
                    medianprops=dict(color="darkblue", linewidth=2),
                    boxprops=dict(linewidth=1.5, color="black"),
                    whiskerprops=dict(linewidth=1.5, color="black"),
                    capprops=dict(linewidth=1.5, color="black"),
                )

                for i, patch in enumerate(bp2["boxes"]):
                    label = global_labels[i]
                    color = CORRECTNESS_COLORS.get(label, "#cccccc")
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(2)

                for i, data in enumerate(global_plot_data):
                    x = np.random.normal(i + 1, 0.05, size=len(data))
                    ax_global.scatter(
                        x, data, alpha=0.35, s=25, c="black", ec="gray", linewidth=0.5
                    )

                ax_global.set_ylabel("Value", fontsize=13, fontweight="bold")
                ax_global.set_title(
                    f"{feature_base} – {stat_label}\ncorrect vs incorrect (all models)",
                    fontsize=13,
                    fontweight="bold",
                    pad=15,
                )
                ax_global.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
                ax_global.spines["top"].set_visible(False)
                ax_global.spines["right"].set_visible(False)
            else:
                ax_global.text(
                    0.5,
                    0.5,
                    "No data for correct/incorrect",
                    transform=ax_global.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="red",
                )
                ax_global.axis("off")

            fig.suptitle(
                f"Classification correctness analysis: {feature_base.replace('_', ' ').title()} – {stat_label}",
                fontsize=16,
                fontweight="bold",
                y=0.97,
            )
            plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.92])

            stats_text = format_influence_statistics_box(x_labels, plot_data)
            add_bottom_stats_panel(fig, ax_models, stats_text, width_frac=0.45, y_margin=0.04)

            if global_plot_data:
                global_stats_text = format_influence_statistics_box(global_labels, global_plot_data)
                add_bottom_stats_panel(fig, ax_global, global_stats_text, width_frac=0.30, y_margin=0.04)

            out_file = (
                feature_folder
                / f"{feature_base}_{stat_label}_correct_vs_incorrect.png"
            )
            plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    print("=" * 80)
    print(f"✅ Saved correct/incorrect boxplots to: {out_dir}")
    print("=" * 80 + "\n")

def viz_features_by_confusion_outcome(df, output_root: Path):
    setup_professional_style()

    out_dir = output_root / "boxplots_TP_FP_TN_FN"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_groups = build_feature_groups(df)
    models = sorted(df["model"].dropna().unique())
    outcomes = ["TP", "FP", "TN", "FN"]

    print("\n" + "=" * 80)
    print("Creating boxplots: features per confusion outcome (TP/FP/TN/FN) ...")
    print(f"Found {len(feature_groups)} feature groups\n")

    for feature_base, columns_list in sorted(feature_groups.items()):
        feature_folder = out_dir / feature_base
        feature_folder.mkdir(parents=True, exist_ok=True)

        stat_order = ["min", "mean", "std", "max"]
        columns_sorted = sorted(
            columns_list,
            key=lambda x: next(
                (i for i, stat in enumerate(stat_order) if stat == x[1]), 999
            ),
        )

        for col, stat in columns_sorted:
            stat_label = stat.upper() if stat != "single" else col

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            ax_models, ax_global = axes

            plot_data = []
            x_labels = []

            for model in models:
                for outc in outcomes:
                    mask = (df["model"] == model) & (df["outcome"] == outc)
                    data = df.loc[mask, col].dropna()
                    if len(data) > 0:
                        plot_data.append(data.values)
                        x_labels.append(f"{model}\n{outc}")

            if not plot_data:
                plt.close(fig)
                continue

            bp = ax_models.boxplot(
                plot_data,
                tick_labels=x_labels,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                vert=True,
                whis=1.5,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=7),
                medianprops=dict(color="darkblue", linewidth=2),
                boxprops=dict(linewidth=1.5, color="black"),
                whiskerprops=dict(linewidth=1.5, color="black"),
                capprops=dict(linewidth=1.5, color="black"),
            )

            for i, patch in enumerate(bp["boxes"]):
                label = x_labels[i]
                outc = None
                for o in outcomes:
                    if o in label:
                        outc = o
                        break
                color = OUTCOME_COLORS.get(outc, "#cccccc")
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor("black")
                patch.set_linewidth(2)

            for i, data in enumerate(plot_data):
                x = np.random.normal(i + 1, 0.05, size=len(data))
                ax_models.scatter(
                    x, data, alpha=0.35, s=25, c="black", ec="gray", linewidth=0.5
                )

            ax_models.set_ylabel("Value", fontsize=13, fontweight="bold")
            ax_models.set_title(
                f"{feature_base} – {stat_label}\nper model (TP/FP/TN/FN)",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            ax_models.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
            ax_models.spines["top"].set_visible(False)
            ax_models.spines["right"].set_visible(False)
            for tick in ax_models.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")

            global_plot_data = []
            global_labels = []

            for outc in outcomes:
                data = df.loc[df["outcome"] == outc, col].dropna()
                if len(data) > 0:
                    global_plot_data.append(data.values)
                    global_labels.append(outc)

            if global_plot_data:
                bp2 = ax_global.boxplot(
                    global_plot_data,
                    tick_labels=global_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=7),
                    medianprops=dict(color="darkblue", linewidth=2),
                    boxprops=dict(linewidth=1.5, color="black"),
                    whiskerprops=dict(linewidth=1.5, color="black"),
                    capprops=dict(linewidth=1.5, color="black"),
                )

                for i, patch in enumerate(bp2["boxes"]):
                    outc = global_labels[i]
                    color = OUTCOME_COLORS.get(outc, "#cccccc")
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(2)

                for i, data in enumerate(global_plot_data):
                    x = np.random.normal(i + 1, 0.05, size=len(data))
                    ax_global.scatter(
                        x, data, alpha=0.35, s=25, c="black", ec="gray", linewidth=0.5
                    )

                ax_global.set_ylabel("Value", fontsize=13, fontweight="bold")
                ax_global.set_title(
                    f"{feature_base} – {stat_label}\nTP/FP/TN/FN (all models)",
                    fontsize=13,
                    fontweight="bold",
                    pad=15,
                )
                ax_global.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
                ax_global.spines["top"].set_visible(False)
                ax_global.spines["right"].set_visible(False)
            else:
                ax_global.text(
                    0.5,
                    0.5,
                    "No data for TP/FP/TN/FN",
                    transform=ax_global.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="red",
                )
                ax_global.axis("off")

            fig.suptitle(
                f"Confusion outcome analysis: {feature_base.replace('_', ' ').title()} – {stat_label}",
                fontsize=16,
                fontweight="bold",
                y=0.97,
            )
            plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.92])

            stats_text = format_influence_statistics_box(x_labels, plot_data)
            add_bottom_stats_panel(fig, ax_models, stats_text, width_frac=0.45, y_margin=0.04)

            if global_plot_data:
                global_stats_text = format_influence_statistics_box(global_labels, global_plot_data)
                add_bottom_stats_panel(fig, ax_global, global_stats_text, width_frac=0.30, y_margin=0.04)

            out_file = feature_folder / f"{feature_base}_{stat_label}_TP_FP_TN_FN.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    print("=" * 80)
    print(f"✅ Saved TP/FP/TN/FN boxplots to: {out_dir}")
    print("=" * 80 + "\n")

def viz_features_vs_prediction_scatter(merged_df, output_root: Path, confidence_threshold=0.3):
    """
    For each feature: scatterplot feature vs prediction_score
    With colors: Confident REAL / Uncertain / Confident FAKE
    + border lines
    """

    setup_professional_style()

    base_folder = output_root / "scatter_prediction"
    base_folder.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print("Creating scatter plots: features vs prediction scores...")
    print("=" * 80 + "\n")

    def categorize_confidence(pred_value):
        if pd.isna(pred_value):
            return "Unknown"
        if abs(pred_value - 0.5) <= confidence_threshold:
            return "Uncertain"
        elif pred_value < 0.5:
            return "Confident REAL"
        else:
            return "Confident FAKE"

    df_plot = merged_df.copy()
    df_plot["prediction_confidence"] = df_plot["prediction_score"].apply(
        categorize_confidence
    )

    CONFIDENCE_COLORS = {
        "Confident REAL": "#1f77b4",
        "Uncertain": "#ff7f0e",
        "Confident FAKE": "#d62728",
        "Unknown": "#7f7f7f",
    }

    CONFIDENCE_ALPHA = {
        "Confident REAL": 0.8,
        "Uncertain": 0.6,
        "Confident FAKE": 0.8,
        "Unknown": 0.3,
    }

    base_exclude = {
        "model",
        "track_id",
        "track_stem",
        "data_type",
        "prediction_score",
        "pred_label",
        "true_label",
        "is_correct",
        "outcome",
        "prediction_confidence",
    }

    all_features = [
        col
        for col in df_plot.columns
        if (
            col not in base_exclude
            and pd.api.types.is_numeric_dtype(df_plot[col])
            and df_plot[col].notna().sum() > 0
        )
    ]

    feature_groups = defaultdict(list)
    for col in all_features:
        parts = col.split("_")
        if len(parts) > 1 and parts[-1] in ["min", "mean", "std", "max"]:
            base_name = "_".join(parts[:-1])
            stat = parts[-1]
        else:
            base_name = col
            stat = "single"
        feature_groups[base_name].append((col, stat))

    print(f"Processing {len(feature_groups)} feature groups...\n")

    for feature_base, columns_list in sorted(feature_groups.items()):
        feature_folder = base_folder / feature_base
        feature_folder.mkdir(exist_ok=True, parents=True)

        stat_order = ["min", "mean", "std", "max"]
        columns_sorted = sorted(
            columns_list,
            key=lambda x: next((i for i, s in enumerate(stat_order) if s == x[1]), 999),
        )

        for col, stat in columns_sorted:

            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            fig.patch.set_facecolor("white")

            plot_data = df_plot[
                [col, "prediction_score", "prediction_confidence"]
            ].dropna()

            if len(plot_data) > 0:
                for confidence in ["Confident REAL", "Uncertain", "Confident FAKE"]:
                    conf_data = plot_data[
                        plot_data["prediction_confidence"] == confidence
                    ]
                    if len(conf_data) > 0:
                        ax.scatter(
                            conf_data[col],
                            conf_data["prediction_score"],
                            alpha=CONFIDENCE_ALPHA[confidence],
                            s=80,
                            color=CONFIDENCE_COLORS[confidence],
                            label=confidence,
                            edgecolors="black",
                            linewidth=0.7,
                        )

                ax.axhline(y=0.5, color="black", linestyle="-", linewidth=2, alpha=0.7)
                ax.axhline(
                    y=0.5 - confidence_threshold,
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Confidence threshold ±{confidence_threshold}",
                )
                ax.axhline(
                    y=0.5 + confidence_threshold,
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                )

                ax.fill_between(
                    ax.get_xlim(),
                    0.5 - confidence_threshold,
                    0.5 + confidence_threshold,
                    color="orange",
                    alpha=0.08,
                )

                ax.set_xlabel(f"{col}", fontsize=13, fontweight="bold")
                ax.set_ylabel("Model Prediction P(Fake)", fontsize=13, fontweight="bold")
                ax.set_title(
                    f"{feature_base.replace('_', ' ').title()} vs Prediction Score",
                    fontsize=14,
                    fontweight="bold",
                    pad=15,
                )
                ax.set_ylim(-0.05, 1.05)

                ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
                ax.set_axisbelow(True)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(1.8)
                ax.spines["bottom"].set_linewidth(1.8)
                ax.legend(fontsize=11, loc="best", framealpha=0.95)

            plt.tight_layout()
            out_file = feature_folder / f"{col}_scatter_analysis.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            print(f"  ✓ {feature_base}/{col}: scatter_analysis.png")

    print("\n" + "=" * 80)
    print("✅ Scatter vs prediction plots created!")
    print(f"   • Output folder: {base_folder}/")
    print("=" * 80 + "\n")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize full-track audio features vs model predictions."
    )
    ap.add_argument(
        "--config",
        default=(str(ROOT / "configs" / "AudioLIME_configs" / "lime_features_vis.yaml")),
        help="Path to the YAML configuration file."
    )
    return ap.parse_args()

def main():
    args = parse_args()
    config = load_yaml(Path(args.config))

    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    features_path = data_cfg.get("features_path", None)
    preds_path = data_cfg.get("predictions_path", None)
    output_root = output_cfg.get("result_path", None)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Features JSON:   {features_path}")
    print(f"Predictions JSON:{preds_path}")
    print(f"Output root:     {output_root}")
    print("=" * 70)

    features_df, _ = load_fulltrack_features(features_path)
    preds_df = load_predictions(preds_path)

    merged_df = merge_features_and_predictions(features_df, preds_df)

    # Feature distribution per model + REAL vs GENERATED
    viz_features_by_model_and_global(merged_df, output_root)

    # Correct vs incorrect classifications distribution
    viz_features_correct_vs_incorrect(merged_df, output_root)

    # Features for TP / FP / TN / FN
    viz_features_by_confusion_outcome(merged_df, output_root)

    # Scatter plots: feature value vs prediction score, colored by confidence
    viz_features_vs_prediction_scatter(merged_df, output_root, confidence_threshold=0.3)

if __name__ == "__main__":
    main()
