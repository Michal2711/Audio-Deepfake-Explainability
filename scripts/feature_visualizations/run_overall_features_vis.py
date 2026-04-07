import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
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

def try_num(s):
    if isinstance(s, bytes):
        s = s.decode('utf-8', errors='ignore')
    match = re.match(r'^(\d+)', s)
    return int(match.group(1)) if match else 999999


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
    Structure features.json:
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
    Structure predictions.json:
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

def plot_predictions_and_features_by_model_line_all(df_features: pd.DataFrame, output_dir: Path):
    """
        Plot predictions and features by model in a line chart.
    """
    setup_professional_style()
    sns.set_theme(style='whitegrid')
    outputdir = output_dir / 'predictions_and_features_line'
    outputdir.mkdir(parents=True, exist_ok=True)
    feature_groups = build_feature_groups(df_features)
    
    feature_cols = [c for c in df_features.columns if pd.api.types.is_numeric_dtype(df_features[c]) and 
                    c not in ['model', 'track_id', 'track_stem', 'data_type', 'prediction_score', 
                              'pred_label', 'true_label', 'is_correct', 'outcome']]
        
    models = sorted(df_features['model'].dropna().unique())
    
    for model in models:
        model_dir = outputdir / model.replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = sorted(df_features[df_features['model'] == model]['track_stem'].dropna().unique(), key=try_num)
        file_idx_map = {f: i for i, f in enumerate(model_files)}
        
        df_model = df_features[df_features['model'] == model].copy()
        df_model['file_index'] = df_model['track_stem'].map(file_idx_map)
        
        d_pred = df_model[['file_index', 'prediction_score']].dropna()
        if d_pred.empty:
            print(f"No predictions for model {model}")
            continue
        d_pred = d_pred.sort_values('file_index').reset_index(drop=True)
        d_pred = d_pred.set_index('file_index').reindex(range(len(model_files)), method='ffill').reset_index()
        d_pred = d_pred.rename(columns={'index': 'file_index'})

        for feature_base, column_line in sorted(feature_groups.items()):
            feature_base = model_dir / feature_base
            feature_base.mkdir(parents=True, exist_ok=True)

            stat_order = ["min", "mean", "std", "max"]
            colums_sorted = sorted(
                column_line,
                key=lambda x: next(
                    (i for i, stat in enumerate(stat_order) if stat == x[1]), 999
                ),
            )

            for col, stat in colums_sorted:
                if col not in df_model.columns:
                    continue
                    
                d_col = df_model[['file_index', col]].dropna()
                if d_col.empty:
                    continue
                
                d_col = d_col.sort_values('file_index').reset_index(drop=True)
                d_col = d_col.set_index('file_index').reindex(range(len(model_files)), method='ffill').reset_index()
                d_col = d_col.rename(columns={'index': 'file_index'})
                
                fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                ax_top.plot(range(len(model_files)), d_pred['prediction_score'], 'o-', linewidth=3, 
                        markersize=7, color='darkred', alpha=0.9, label='Prediction Score')
                ax_top.set_ylabel('Prediction Score\n(P > 0.5 = Fake)', fontsize=12, fontweight='bold')
                ax_top.axhline(y=0.5, color='black', linestyle='-', linewidth=2, alpha=0.8)
                ax_top.grid(True, alpha=0.3, linestyle='--')
                ax_top.legend(loc='upper right')
                
                ax_bottom.plot(range(len(model_files)), d_col[col], 'o-', linewidth=3, markersize=7, 
                            color='steelblue', alpha=0.8, label=col.replace('_', ' ').title())
                ax_bottom.set_xlabel('File Index (sorted by name)')
                ax_bottom.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
                ax_bottom.grid(True, alpha=0.3, linestyle='--')
                ax_bottom.legend(loc='upper right')
                
                ax_bottom.set_title(f'{model}: {col.replace("_", " ").title()} vs Predictions', fontsize=13, pad=20)
                
                short_labels = [f"{i:2d}: {f[:20]}..." if len(f) > 20 else f"{i:2d}: {f}" 
                            for i, f in enumerate(model_files)]
                fig.text(0.92, 0.5, 'File Mapping:\n' + '\n'.join(short_labels), 
                        fontsize=13, va='center', ha='left',
                        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.95))
                
                safe_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                outfile = feature_base / f"{safe_col}_with_predictions.png"
                plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"Saved: {model}/{safe_col}_with_predictions.png")

def plot_features_by_model_line_all(df_features: pd.DataFrame, output_dir: Path):
    """
        Visualization for physical features per model for ALL files.
    """
    setup_professional_style()
    sns.set_theme(style="whitegrid")

    output_dir = output_dir / "lineplots_by_model_all_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_groups = build_feature_groups(df_features)
    feature_cols = [c for c in df_features.columns 
                   if pd.api.types.is_numeric_dtype(df_features[c]) and 
                   c not in ['model', 'track_id', 'track_stem', 'data_type', 'prediction_score', 'is_correct']]
    
    for model in sorted(df_features['model'].dropna().unique()):
        model_dir = output_dir / model.replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = sorted(df_features[df_features['model'] == model]['track_stem'].dropna().unique(), key=try_num)
        file_idx_map = {f: i for i, f in enumerate(model_files)}
        
        df_model = df_features[df_features['model'] == model].copy()
        df_model['file_index'] = df_model['track_stem'].map(file_idx_map)
        
        for feature_base, column_list in sorted(feature_groups.items()):
            feature_base = model_dir / feature_base
            feature_base.mkdir(parents=True, exist_ok=True)

            stat_order = ["min", "mean", "std", "max"]
            colums_sorted = sorted(
                column_list,
                key=lambda x: next(
                    (i for i, stat in enumerate(stat_order) if stat == x[1]), 999
                ),
            )
            
            for col, stat in colums_sorted:
                dcol = df_model[['file_index', col]].dropna()
                if dcol.empty:
                    continue
                
                dcol = dcol.sort_values('file_index').reset_index(drop=True)
                dcol = dcol.set_index('file_index').reindex(range(len(model_files)), method='ffill').reset_index()
                dcol = dcol.rename(columns={'index': 'file_index'})
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(range(len(model_files)), dcol[col], 'o-', linewidth=2.5, markersize=6, alpha=0.8, color='steelblue')
                
                ax.set_xlabel("File index")
                ax.set_ylabel(col.replace('_', ' ').title())
                ax.set_title(f"{model}: {col.replace('_', ' ').title()}", fontsize=13)
                ax.set_xticks(range(len(model_files)))  # pełne 0-N
                ax.grid(True, alpha=0.3, linestyle='--')
                
                short_labels = [f[:18] + "..." if len(f) > 18 else f for f in model_files]
                index_text = "\n".join(f"{i:2d}: {lab}" for i, lab in enumerate(short_labels))
                fig.text(1.01, 0.5, f"File Mapping ({model}):\n{index_text}",
                        fontsize=8.8, va="center", ha="left",
                        bbox=dict(facecolor="white", edgecolor="#d1d5db", boxstyle="round,pad=0.4", alpha=0.95))
                
                plt.subplots_adjust(right=0.78, bottom=0.15)
                fig.tight_layout()
                
                safe_feature = re.sub(r'[^a-zA-Z0-9_]', '_', col)[:40]
                outfile = feature_base / f"{safe_feature}_by_file_line.png"
                plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"💾 {model}/{col}")
    
    print(f"✅ Features by model saved to {output_dir}")

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

MODEL_ORDER = ['ElevenLabs', 'REAL', 'SUNO', 'SUNO_PRO', 'UDIO']

TYPE_MAPPING = {
    'ElevenLabs': 'GENERATED',
    'REAL':       'REAL',
    'SUNO':       'GENERATED',
    'SUNO_PRO':   'GENERATED',
    'UDIO':       'GENERATED',
}

FEATURE_GROUPS_DEF = {
    'Signal_energy': [
        'rms_'
    ],
    'Frequency_spectrum': [
        'spectral_'
    ],
    'Fundamental_Frequency_Pitch': [
        'f0_', 'intonation_'
    ],
    'Jitter_Shimmer': [
        'jitter_', 'shimmer_'
    ],
    'Vocal_quality': [
        'hnr', 'voice_breaks', 'breath_count'
    ],
    'Rhythm_and_temporal_features': [
        'zero_crossing_rate', 'rhythm_'
    ]
}

_CORR_EXCLUDE = {
    'model', 'track', 'track_id', 'data_type', 'data_type_str',
    'component_name', 'component_type', 'component_key',
    'prediction_score', 'predicted_class',
    'vocals0_influence', 'drums0_influence', 'bass0_influence', 'other0_influence',
    'abs_importance', 'importance', 'component', 'track_stem',
    'low_freq', 'high_freq', 'band_type',
}

TBL_BG         = '#0e1117'
TBL_HEADER_BG  = '#1a1d27'
TBL_ROW_ALT_BG = '#13161f'
TBL_TEXT       = '#d0d0d0'
TBL_HEADER_TXT = '#7a8099'
TBL_POS_STRONG = '#ff6b35'
TBL_POS_MEDIUM = '#e8943a'
TBL_NEG_STRONG = '#2ecc71'
TBL_NEG_MEDIUM = '#27ae60'
TBL_NEAR_ZERO  = '#8899aa'

def assign_feature_group_overall(col: str) -> str:
    for group, prefixes in FEATURE_GROUPS_DEF.items():
        for prefix in prefixes:
            if col.startswith(prefix):
                return group
    return 'other'

def _build_corr_matrix_overall(df, feature_cols, target_col, groups_bool):
    stat_order  = {'mean': 0, 'std': 1, 'min': 2, 'max': 3}
    stat_suffix = re.compile(r'_(mean|std|min|max)$')

    rdict = {}
    for label, mask in groups_bool.items():
        gdf   = df[mask]
        rvals = {}
        for feat in feature_cols:
            sub = gdf[[feat, target_col]].dropna()
            rvals[feat] = sub[feat].corr(sub[target_col]) if len(sub) >= 3 else np.nan
        rdict[label] = rvals

    rdf = pd.DataFrame(rdict).dropna(how='all')
    if rdf.empty:
        return rdf

    def _base(c): return stat_suffix.sub('', c)
    def _rank(c):
        m = stat_suffix.search(c)
        return stat_order.get(m.group(1), 99) if m else -1

    rdf['_base'] = [_base(c) for c in rdf.index]
    rdf['_bimp'] = (rdf.drop(columns=['_base']).abs().max(axis=1)
                    .groupby(rdf['_base']).transform('max'))
    rdf['_rank'] = [_rank(c) for c in rdf.index]
    rdf = (rdf.sort_values(['_bimp', '_base', '_rank'],
                           ascending=[False, True, True])
              .drop(columns=['_base', '_bimp', '_rank']))
    return rdf

def _save_corr_heatmap_overall(rdf, title, out_file):
    if rdf.empty:
        print(f'  [SKIP] Empty matrix → {out_file.name}')
        return
    n_feats = len(rdf)
    n_cols  = len(rdf.columns)
    fig_h   = max(4, n_feats * 0.42 + 2.5)
    fig_w   = max(10, n_cols * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    mask_nan = rdf.isnull()
    sns.heatmap(
        rdf, ax=ax, cmap='coolwarm', vmin=-1, vmax=1,
        annot=True, fmt='.2f', linewidths=0.4, linecolor='#dddddd',
        mask=mask_nan,
        cbar_kws={'label': 'Pearson r', 'shrink': 0.6},
        annot_kws={'size': 8, 'weight': 'bold'},
    )
    ax.patch.set_facecolor('#f0f0f0')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=10)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out_file.name}')

def plot_overall_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    model_order: list = None,
) -> None:
    """
    For each semantic feature group, generate a Pearson r heatmap:
        feature vs predictionscore

    Columns: all | generated | real | ElevenLabs | REAL | SUNO | SUNO_PRO | UDIO
    Rows: feature names, sorted in descending order by max|r|
    Color:   coolwarm, range –1 … +1

    Output structure:
        output_dir/overall_correlation_r_heatmaps/
            Signal energy/  Signal energy_r_vs_prediction.png
            ...
            allfeatures_r_vs_prediction.png
    """
    setup_professional_style()
    sns.set_theme(style='whitegrid')

    if model_order is None:
        model_order = MODEL_ORDER

    root_out = output_dir / 'overall_correlation_r_heatmaps'
    root_out.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    feat_cols    = [
        c for c in df.columns
        if c not in _CORR_EXCLUDE
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().sum() > 0
    ]
    feat_to_grp  = {c: assign_feature_group_overall(c) for c in feat_cols}
    groups_present = sorted(set(feat_to_grp.values()))

    group_defs = {
        'all':       pd.Series(True, index=df.index),
        'generated': df['data_type'] == 'GENERATED',
        'real':      df['data_type'] == 'REAL',
        **{m: df['model'] == m for m in model_order if m in df['model'].unique()},
    }

    target_col = 'prediction_score'
    if target_col not in df.columns:
        print('[WARN] Column "prediction_score" not found in DataFrame → skipping overall correlation heatmaps')
        return

    print('─' * 70)
    print('Genearating OVERALL heatmap correlation r')
    print(f'  Feature groups: {groups_present}')
    print('─' * 70)

    for feat_group in groups_present:
        grp_feats = [
            c for c, g in feat_to_grp.items()
            if g == feat_group and df[c].notna().sum() >= 3
        ]
        if not grp_feats:
            continue

        grp_dir = root_out / feat_group
        grp_dir.mkdir(parents=True, exist_ok=True)

        rdf   = _build_corr_matrix_overall(df, grp_feats, target_col, group_defs)
        title = f'{feat_group}  –  Pearson r vs Prediction P(fake)'
        out_f = grp_dir / f'{feat_group}_r_vs_prediction.png'
        _save_corr_heatmap_overall(rdf, title, out_f)

    all_feats = [c for c in feat_cols if df[c].notna().sum() >= 3]
    rdf_all   = _build_corr_matrix_overall(df, all_feats, target_col, group_defs)
    if not rdf_all.empty and 'all' in rdf_all.columns:
        rdf_all = rdf_all.reindex(
            rdf_all['all'].abs().sort_values(ascending=False).index
        )
    _save_corr_heatmap_overall(
        rdf_all,
        'All features  –  Pearson r vs Prediction P(fake)',
        root_out / 'all_features_r_vs_prediction.png',
    )

    print(f'Overall correlation heatmaps → {root_out}')
    print('─' * 70)


def _tbl_fmt_value(v):
    if pd.isna(v): return ''
    a = abs(v)
    if a == 0:     return '0'
    if a >= 1000:  return f'{v:,.0f}'
    if a >= 10:    return f'{v:.2f}'
    if a >= 1:     return f'{v:.3f}'
    if a >= 0.001: return f'{v:.4f}'
    return f'{v:.2e}'


def _tbl_fmt_pct(pct):
    if pd.isna(pct) or abs(pct) < 5: return ''
    sign = '+' if pct > 0 else ''
    return f'{sign}{pct:.0f}%'


def _tbl_pct_color(pct):
    if pd.isna(pct) or abs(pct) < 5: return TBL_NEAR_ZERO
    if pct > 0: return TBL_POS_STRONG if abs(pct) >= 30 else TBL_POS_MEDIUM
    return TBL_NEG_STRONG if abs(pct) >= 30 else TBL_NEG_MEDIUM


def _draw_overall_table(feat_list, real_vals, means_v, pct_df, sources,
                        title_str, out_file,
                        figsize_w=14.0, row_height=0.40, dpi=180,
                        col_header_colors=None, strip_stat_suffix=True):
    import matplotlib.patches as mpatches

    n_rows = len(feat_list)
    n_cols = 2 + len(sources)
    fig_h  = max(4.0, n_rows * row_height + 1.8)
    fig    = plt.figure(figsize=(figsize_w, fig_h), facecolor=TBL_BG)
    ax     = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(TBL_BG)
    ax.axis('off')

    col_labels = ['FEATURE', 'REAL'] + sources
    raw_widths = [0.30] + [0.12] * (n_cols - 1)
    tot_w      = sum(raw_widths)
    col_widths = [w / tot_w for w in raw_widths]
    col_lefts  = []
    x = 0.01
    for w in col_widths:
        col_lefts.append(x)
        x += w * 0.99 / tot_w

    def cell(r_idx, c_idx, text, color=TBL_TEXT, bg=TBL_BG,
             fs=8.5, bold=False, align='right'):
        x0 = col_lefts[c_idx]
        cw = col_widths[c_idx]
        y0 = 1.0 - (r_idx + 1) * (1.0 / (n_rows + 2))
        ch = 1.0 / (n_rows + 2)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), cw, ch,
            boxstyle='square,pad=0', linewidth=0, 
            facecolor=bg, transform=ax.transAxes, clip_on=False,
        ))
        tx = x0 + cw * 0.95 if align == 'right' else x0 + cw * 0.05
        ax.text(tx, y0 + ch * 0.5, text,
                color=color, fontsize=fs, ha=align, va='center',
                fontweight='bold' if bold else 'normal',
                transform=ax.transAxes, clip_on=False,
                fontfamily='monospace')

    for ci, lbl in enumerate(col_labels):
        hdr_color = (col_header_colors or {}).get(lbl, TBL_HEADER_TXT)
        cell(0, ci, lbl.upper().replace('_', ' '),
             color=hdr_color, bg=TBL_HEADER_BG, fs=8, bold=True,
             align='left' if ci == 0 else 'right')

    for ri, feat in enumerate(feat_list, start=1):
        row_bg = TBL_ROW_ALT_BG if ri % 2 == 0 else TBL_BG
        disp   = re.sub(r'_(mean|std|min|max)$', '', feat) if strip_stat_suffix else feat
        disp   = disp.replace('_', ' ').title()
        cell(ri, 0, disp, color=TBL_TEXT, bg=row_bg, align='left')
        real_v = real_vals[feat] if feat in real_vals.index else np.nan
        cell(ri, 1, _tbl_fmt_value(real_v), color=TBL_TEXT, bg=row_bg)
        for si, src in enumerate(sources):
            sv  = means_v.loc[src, feat] if src in means_v.index else np.nan
            pct = pct_df.loc[feat, src]  if src in pct_df.columns else np.nan
            txt = f'{_tbl_fmt_value(sv)} {_tbl_fmt_pct(pct)}'
            cell(ri, 2 + si, txt, color=_tbl_pct_color(pct), bg=row_bg)

    ax.text(0.01, 0.995, title_str,
            color='#aabbcc', fontsize=9.5, fontweight='bold',
            ha='left', va='top', transform=ax.transAxes, fontfamily='monospace')

    legend = [
        (TBL_POS_STRONG, '≥+30%'), (TBL_POS_MEDIUM, '+15-30%'),
        (TBL_NEAR_ZERO,  '≈0%'),
        (TBL_NEG_MEDIUM, '−15-30%'), (TBL_NEG_STRONG, '≥−30%'),
    ]
    ax.text(0.01, 0.008, 'Deviation from REAL: ',
            color=TBL_HEADER_TXT, fontsize=7, ha='left', va='bottom',
            transform=ax.transAxes)
    lx = 0.17
    for col, lbl in legend:
        ax.text(lx, 0.008, f' {lbl}', color=col, fontsize=7,
                ha='left', va='bottom', transform=ax.transAxes,
                fontfamily='monospace')
        lx += 0.10

    plt.savefig(out_file, dpi=dpi, bbox_inches='tight',
                facecolor=TBL_BG, edgecolor='none')
    plt.close()
    print(f'  Saved: {out_file.name}')


def _build_pred_split_overall(df, feat_cols, sources):
    compound, col_colors, rows = [], {}, {}
    if 'pred_label' not in df.columns:
        return pd.DataFrame(), [], {}
    for src in sources:
        src_df = df[df['model'] == src]
        for pred_lbl, color in [('Real', TBL_NEG_MEDIUM), ('Fake', TBL_POS_STRONG)]:
            key = f'{src} {pred_lbl}'
            compound.append(key)
            col_colors[key] = color
            subset = src_df[src_df['pred_label'] == pred_lbl]
            rows[key] = (subset[feat_cols].mean()
                         if not subset.empty
                         else pd.Series(np.nan, index=feat_cols))
    means_split = pd.DataFrame(rows).T
    return means_split, compound, col_colors

def plot_overall_comparison_table(
    df: pd.DataFrame,
    output_dir: Path,
    model_order: list = None,
    feature_groups: dict = None,
    multi_stat_groups: list = None,
    sort_by_deviation: bool = True,
    figsize_w: float = 14.0,
    row_height: float = 0.40,
    dpi: int = 180,
) -> None:
    """
    For each semantic feature group, generate a dark-themed PNG comparison table.

    Rows: features names, sorted by max|% deviation from REAL
    Columns: REAL (baseline) | ElevenLabs | SUNO | SUNOPRO | UDIO
    Color:   deviation from REAL (green = less than REAL, orange = more than REAL)

    Output structure:
        output_dir/overall_comparison_tables/
            SignalEnergy/     Signal energy.png   + Signal energy_by_pred.png
            FrequencySpectrum/
                Frequency spectrum_mean.png
                Frequency spectrum_mean_by_pred.png
                ...
            ...
            all_features.png
    """
    setup_professional_style()

    if model_order    is None: model_order    = MODEL_ORDER
    if feature_groups is None: feature_groups = FEATURE_GROUPS_DEF
    if multi_stat_groups is None: multi_stat_groups = ['Frequency spectrum']

    root_out = output_dir / 'overall_comparison_tables'
    root_out.mkdir(parents=True, exist_ok=True)

    meta_cols = {
        'model', 'track_id', 'track_id_pred', 'track_stem', 'data_type',
        'prediction_score', 'pred_label', 'true_label', 'is_correct', 'outcome',
    }
    all_feat_cols = [
        c for c in df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    stat_re         = re.compile(r'_(mean|std|min|max)$')
    stat_order_list = ['mean', 'std', 'min', 'max']

    def feat_group(col):
        for g, prefixes in feature_groups.items():
            for p in prefixes:
                if col.startswith(p):
                    return g
        return 'other'

    means = df.groupby('model')[all_feat_cols].mean()
    if 'REAL' not in means.index:
        print('[WARN] Baseline "REAL" not found in data → cannot compute deviations → skipping overall comparison tables')
        return

    real_vals = means.loc['REAL']
    sources   = [m for m in model_order if m in means.index and m != 'REAL']

    with np.errstate(divide='ignore', invalid='ignore'):
        pct_diffs = {}
        for src in sources:
            sv  = means.loc[src]
            pct = np.where(real_vals != 0,
                           (sv - real_vals) / real_vals.abs() * 100,
                           np.nan)
            pct_diffs[src] = pd.Series(pct, index=all_feat_cols)
    pct_df_full = pd.DataFrame(pct_diffs)

    valid       = real_vals.dropna().index
    real_vals   = real_vals.loc[valid]
    means_v     = means[valid]
    pct_df_full = pct_df_full.loc[valid]

    def _type_split(feat_cols):
        has_pred = 'pred_label' in df.columns
        masks = {'GENERATED': df['data_type'] == 'GENERATED'}
        colors = {'GENERATED': TBL_HEADER_TXT}
        if has_pred:
            masks['GEN → pred Real'] = (
                (df['data_type'] == 'GENERATED') & (df['pred_label'] == 'Real')
            )
            masks['GEN → pred Fake'] = (
                (df['data_type'] == 'GENERATED') & (df['pred_label'] == 'Fake')
            )
            colors['GEN → pred Real'] = TBL_NEG_MEDIUM
            colors['GEN → pred Fake'] = TBL_POS_STRONG

        rows = {}
        for label, mask in masks.items():
            subset = df[mask]
            rows[label] = (subset[feat_cols].mean()
                           if not subset.empty
                           else pd.Series(np.nan, index=feat_cols))

        mv = pd.DataFrame(rows).T
        srcs = list(masks.keys())

        with np.errstate(divide='ignore', invalid='ignore'):
            pd_diffs = {}
            for src in srcs:
                sv  = mv.loc[src]
                pct = np.where(
                    real_vals[feat_cols] != 0,
                    (sv - real_vals[feat_cols]) / real_vals[feat_cols].abs() * 100,
                    np.nan,
                )
                pd_diffs[src] = pd.Series(pct, index=feat_cols)
        pct = pd.DataFrame(pd_diffs)

        return mv, pct, srcs, colors

    def _pred_split_pct(ms, cs, feat_cols):
        with np.errstate(divide='ignore', invalid='ignore'):
            return pd.DataFrame({
                c: np.where(
                    real_vals[feat_cols] != 0,
                    (ms.loc[c] - real_vals[feat_cols])
                    / real_vals[feat_cols].abs() * 100,
                    np.nan,
                ) if c in ms.index else np.nan
                for c in cs
            }, index=feat_cols)

    all_groups = list(feature_groups.keys()) + ['other']

    print('─' * 70)
    print('Generating OVERALL comparison tables (without splitting by components)')
    print('─' * 70)

    for grp in all_groups:
        grp_feats = [c for c in valid if feat_group(c) == grp]
        if not grp_feats:
            continue

        grp_dir = root_out / grp.replace(' ', '')
        grp_dir.mkdir(parents=True, exist_ok=True)

        if grp in multi_stat_groups:
            for stat in stat_order_list:
                stat_feats = [c for c in grp_feats if c.endswith(f'_{stat}')]
                if not stat_feats:
                    continue
                if sort_by_deviation:
                    stat_feats = list(
                        pct_df_full.loc[stat_feats].abs()
                        .max(axis=1).sort_values(ascending=False).index
                    )
                title = f'OVERALL  {grp} [{stat.upper()}]  –  mean vs baseline REAL'
                _draw_overall_table(
                    stat_feats, real_vals, means_v, pct_df_full, sources,
                    title, grp_dir / f'{grp}_{stat}.png',
                    figsize_w=figsize_w, row_height=row_height, dpi=dpi,
                )
                ms, cs, chc = _build_pred_split_overall(df, stat_feats, sources)
                if not ms.empty:
                    _draw_overall_table(
                        stat_feats, real_vals, ms, 
                        _pred_split_pct(ms, cs, stat_feats), cs,
                        f'{title} – decomposition by prediction',
                        grp_dir / f'{grp}_{stat}_by_pred.png',
                        figsize_w=figsize_w * 1.6, row_height=row_height,
                        dpi=dpi, col_header_colors=chc,
                    )

                tv_means, tv_pct, tv_sources, tv_colors = _type_split(stat_feats)
                _draw_overall_table(
                    stat_feats, real_vals, tv_means, tv_pct, tv_sources,
                    f'OVERALL  {grp} [{stat.upper()}]  –  sample type vs REAL',
                    grp_dir / f'{grp}_{stat}_by_type.png',
                    figsize_w=figsize_w, row_height=row_height, dpi=dpi,
                    col_header_colors=tv_colors,
                )
        else:
            if sort_by_deviation:
                grp_feats = list(
                    pct_df_full.loc[grp_feats].abs()
                    .max(axis=1).sort_values(ascending=False).index
                )
            title = f'OVERALL  {grp}  –  mean vs baseline REAL'
            _draw_overall_table(
                grp_feats, real_vals, means_v, pct_df_full, sources,
                title, grp_dir / f'{grp}.png',
                figsize_w=figsize_w, row_height=row_height, dpi=dpi,
                strip_stat_suffix=False,
            )
            ms, cs, chc = _build_pred_split_overall(df, grp_feats, sources)
            if not ms.empty:
                _draw_overall_table(
                    grp_feats, real_vals, ms,
                    _pred_split_pct(ms, cs, grp_feats), cs,
                    f'{title} – decomposition by prediction',
                    grp_dir / f'{grp}_by_pred.png',
                    figsize_w=figsize_w * 1.6, row_height=row_height,
                    dpi=dpi, col_header_colors=chc, strip_stat_suffix=False,
                )
            tv_means, tv_pct, tv_sources, tv_colors = _type_split(grp_feats)
            _draw_overall_table(
                grp_feats, real_vals, tv_means, tv_pct, tv_sources,
                f'OVERALL  {grp}  –  sample type vs REAL',
                grp_dir / f'{grp}_by_type.png',
                figsize_w=figsize_w, row_height=row_height, dpi=dpi,
                col_header_colors=tv_colors, strip_stat_suffix=False,
            )

    all_valid = list(valid)
    if sort_by_deviation:
        all_valid = list(
            pct_df_full.abs().max(axis=1).sort_values(ascending=False).index
        )
    _draw_overall_table(
        all_valid, real_vals, means_v, pct_df_full, sources,
        'OVERALL  All features  –  mean vs baseline REAL',
        root_out / 'all_features.png',
        figsize_w=figsize_w, row_height=row_height, dpi=dpi,
        strip_stat_suffix=False,
    )

    ms, cs, chc = _build_pred_split_overall(df, all_valid, sources)
    if not ms.empty:
        _draw_overall_table(
            all_valid, real_vals, ms,
            _pred_split_pct(ms, cs, all_valid), cs,
            'OVERALL  All features  –  decomposition by prediction',
            root_out / 'all_features_by_pred.png',
            figsize_w=figsize_w * 1.6, row_height=row_height, dpi=dpi,
            col_header_colors=chc, strip_stat_suffix=False,
        )

    tv_means, tv_pct, tv_sources, tv_colors = _type_split(all_valid)
    _draw_overall_table(
        all_valid, real_vals, tv_means, tv_pct, tv_sources,
        'OVERALL  All features  –  sample type vs REAL',
        root_out / 'all_features_by_type.png',
        figsize_w=figsize_w, row_height=row_height, dpi=dpi,
        col_header_colors=tv_colors, strip_stat_suffix=False,
    )

    print(f'Overall comparison tables → {root_out}')
    print('─' * 70)


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
    # viz_features_by_model_and_global(merged_df, output_root)

    # Correct vs incorrect classifications distribution - TODO - maybe not so useful
    # viz_features_correct_vs_incorrect(merged_df, output_root)

    # Features for TP / FP / TN / FN
    # viz_features_by_confusion_outcome(merged_df, output_root)

    # Scatter plots: feature value vs prediction score, colored by confidence
    # viz_features_vs_prediction_scatter(merged_df, output_root, confidence_threshold=0.3)

    # plot_features_by_model_line_all(merged_df, output_root) - maybe not so useful

    # plot_predictions_and_features_by_model_line_all(merged_df, output_root)

    # plot_overall_correlation_heatmap(merged_df, output_root)

    plot_overall_comparison_table(merged_df, output_root)

if __name__ == "__main__":
    main()
