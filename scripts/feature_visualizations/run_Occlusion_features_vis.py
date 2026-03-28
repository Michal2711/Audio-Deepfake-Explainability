import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROFESSIONAL_COLORS = {
    'REAL': '#1f77b4',
    'ElevenLabs': '#ff7f0e',
    'SUNO': '#2ca02c',
    'SUNO_PRO': '#d62728',
    'UDIO': '#9467bd'
}

BOX_FILL_COLORS = {
    'REAL': '#aec7e8',
    'ElevenLabs': '#ffbb78',
    'SUNO': '#98df8a',
    'SUNO_PRO': '#ff9896',
    'UDIO': '#c5b0d5'
}

SIGN_COLORS = {
    'positive': '#2ca02c',
    'negative': '#d62728'
}

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize occlusion feature importance results"
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "occlusion_features_vis.yaml"),
        help="Path to the YAML configuration file",
    )
    return ap.parse_args()

def flatten_feature(feat_dict, prefix=''):
                    result = {}
                    
                    for key, val in feat_dict.items():
                        col_name = f"{prefix}_{key}" if prefix else key
                        
                        if isinstance(val, dict):
                            stats_keys = {'min', 'mean', 'std', 'max'}
                            
                            if stats_keys.intersection(val.keys()):
                                for stat_name, stat_val in val.items():
                                    result[f"{col_name}_{stat_name}"] = float(stat_val) if isinstance(stat_val, (int, float)) else np.nan
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
                            try:
                                result[col_name] = float(val)
                            except (ValueError, TypeError):
                                pass
                    
                    return result

def load_and_prepare_data_full(json_file):
    """
    Load JSON data and preserve ALL sub-features from nested structure.
    
    Data structure example:
    {
        model_name: {
            track_id: {
                "type": "patch",
                "patches": {
                    patch_name: {
                        "features": {
                            "duration": 120.0,
                            "rms_wave": {"min": ..., "mean": ..., "std": ..., "max": ...},
                            "jitter": {"jitter_local": ..., "jitter_rap": ..., ...},
                            ...
                        },
                        "occlusion_meta": {
                            "group": "words" | "best" | "most_influential",
                            "rank": 1,
                            "importance": -0.1234,
                            "abs_importance": 0.1234,
                            "tstart": 0,
                            "tend": 512,
                            "fstart": 0,
                            "fend": 512,
                            "start_time_sec": 0.0,
                            "end_time_sec": 5.944308390022676,
                            "patch_type": NEGATIVE | POSITIVE,
                            "model": "ElevenLabs" | "REAL" | "SUNO" | "SUNO_PRO" | "UDIO",
                            "track_stem": "some_track_name"
                        }
                    ...
                    }
                }
            },
            ...
        },
        ...
    }
    
    Output:
    - DataFrame with collumns: model, track, data_type, source, segment_id, [all_features}]
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_rows = []
    
    type_mapping = {
        'ElevenLabs': 'GENERATED',       
        'REAL': 'REAL',
        'SUNO': 'GENERATED',
        'SUNO_PRO': 'GENERATED',
        'UDIO': 'GENERATED',
    }
    
    for model_name, tracks_dict in data.items():
        for track_key, track_data in tracks_dict.items():
            
            if not isinstance(track_data, dict) or 'patches' not in track_data:
                continue
            
            patches_root = track_data.get('patches', {})
            track_type = track_data.get('type', 'unknown')

            for patch_key, patch_data in patches_root.items():

                if not isinstance(patch_data, dict) or 'features' not in patch_data:
                    continue

                features = patch_data.get('features', {})
                occlusion_meta = patch_data.get('occlusion_meta', {})
                            
                row = {
                    'model': model_name,
                    'track': track_key,
                    'patch_key': patch_key,
                    'data_type': type_mapping.get(model_name, model_name),
                    **flatten_feature(occlusion_meta)
                }
                
                flattened = flatten_feature(features)
                row.update(flattened)
                
                all_rows.append(row)
    
    features_df = pd.DataFrame(all_rows)
    
    if features_df.empty:
        print("⚠️ Warning: No data loaded from JSON file!")
        return features_df, []
    
    exclude_cols = {'model', 'track', 'patch_key', 'data_type'}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"\n{'='*80}")
    print(f"✅ Data loaded successfully!")
    print(f"   • Models: {features_df['model'].unique().tolist()}")
    print(f"   • Total records: {len(features_df)}")
    print(f"   • Total features: {len(feature_cols)}")
    print(f"   • Sample features: {feature_cols[:10]}")
    print(f"{'='*80}\n")
    
    return features_df, feature_cols

def format_influence_statistics_box(labels, plot_data):
    rows = []

    header = ["Group", "Mean", "Std", "Count"]
    rows.append(header)

    for label, data in zip(labels, plot_data):
        if data is None or len(data) == 0:
            continue

        if "\n" in label:
            model, sign = label.split("\n", 1)
            group_name = f"{model} ({sign})"
        else:
            group_name = label

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
        return "  ".join(cells)

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
        0.0, 1.0, text,
        ha="left", va="top",
        fontsize=9,
        transform=stats_ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.95,
            edgecolor="black",
            linewidth=1.0
        ),
        family="monospace"
    )

    return stats_ax

def add_group_from_patch_key(features_df):
    df = features_df.copy()

    df['patch_key'] = df['patch_key'].astype(str)

    conditions = [
        df['patch_key'].str.contains('most_influential', case=False, na=False),
        df['patch_key'].str.contains('best', case=False, na=False),
        df['patch_key'].str.contains('worst', case=False, na=False),
    ]
    choices = ['most_influential', 'best', 'worst']

    df['group'] = np.select(conditions, choices, default='other')

    return df

def add_window_label_from_patch_meta(features_df):
    df = features_df.copy()

    required_cols = ["tstart", "tend", "fstart", "fend"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping window labels.")
            return df

    df["t_start"] = df["tstart"].round(1)
    df["t_end"] = df["tend"].round(1)
    df["f_start"] = df["fstart"].round(0).astype(int)
    df["f_end"] = df["fend"].round(0).astype(int)

    df["window_label"] = (
        df["t_start"].astype(str)
        + "-"
        + df["t_end"].astype(str)
        + "s_"
        + df["f_start"].astype(str)
        + "-"
        + df["f_end"].astype(str)
        + "Mel"
    )

    print(f"Added window_label: {df['window_label'].nunique()} unique windows")
    print("Sample windows:", df["window_label"].value_counts().head().to_dict())
    return df


def setup_professional_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    
    sns.set_palette("husl")

def viz2_real_vs_generated_boxplots_with_influence(
        features_df,
        base_output_folder=Path('./'),
        importance_col='importance'
):
    setup_professional_style()

    base_folder = Path(f'{base_output_folder}/visualizations_boxplot_pos_vs_neg')
    base_folder.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*80}")
    print("Creating INFLUENCE-SPLIT boxplot visualizations for academic thesis...")
    print(f"{'='*80}\n")

    if importance_col not in features_df.columns:
        raise ValueError(f"Column '{importance_col}' not found in features_df")

    df = features_df.copy()
    df['influence_sign'] = np.where(df[importance_col] >= 0,
                                    'positive', 'negative')

    exclude_cols = {
        'model', 'track', 'patch_key', 'data_type',
        importance_col, 'influence_sign'
    }

    occlusion_meta_cols = {
        'group', 'rank', 'importance', 'abs_importance',
        'tstart', 'tend', 'fstart', 'fend',
        'start_time_sec', 'end_time_sec',
        'patch_type', 'model', 'track_stem'
    }

    all_cols = [
        col for col in df.columns
        if col not in exclude_cols and col not in occlusion_meta_cols
    ]

    feature_groups = defaultdict(list)

    for col in all_cols:
        parts = col.split('_')
        if len(parts) > 1 and parts[-1] in ['min', 'mean', 'std', 'max']:
            base_name = '_'.join(parts[:-1])
            stat = parts[-1]
        else:
            base_name = col
            stat = 'single'
        feature_groups[base_name].append((col, stat))

    print(f"Found {len(feature_groups)} feature groups\n")

    models = sorted(df['model'].dropna().unique())
    signs = ['negative', 'positive']

    for feature_base, columns_list in sorted(feature_groups.items()):
        print(f"Processing feature: {feature_base}")

        feature_folder = base_folder / feature_base
        feature_folder.mkdir(exist_ok=True, parents=True)

        if len(columns_list) == 1 and columns_list[0][1] == 'single':
            col = columns_list[0][0]

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            ax_models, ax_global = axes

            plot_data = []
            x_labels = []

            for model in models:
                for sign in signs:
                    mask = (
                        (df['model'] == model) &
                        (df['influence_sign'] == sign)
                    )
                    data = df.loc[mask, col].dropna()
                    if len(data) > 0:
                        plot_data.append(data.values)
                        x_labels.append(f'{model}\n{sign}')
                    else:
                        plot_data.append(None)
                        x_labels.append(f'{model}\n{sign}')

            non_empty_indices = [
                i for i, d in enumerate(plot_data)
                if d is not None and len(d) > 0
            ]
            if not non_empty_indices:
                print(f"  ⚠️ SKIPPED: No valid data for feature '{col}'")
                plt.close(fig)
                continue

            plot_data = [plot_data[i] for i in non_empty_indices]
            x_labels = [x_labels[i] for i in non_empty_indices]

            bp = ax_models.boxplot(
                plot_data,
                tick_labels=x_labels,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                meanline=False,
                notch=False,
                vert=True,
                whis=1.5,
                meanprops=dict(
                    marker='D', markerfacecolor='red',
                    markersize=7, markeredgecolor='darkred',
                    markeredgewidth=1.5
                ),
                medianprops=dict(color='darkblue', linewidth=2),
                whiskerprops=dict(linewidth=1.5, color='black'),
                capprops=dict(linewidth=1.5, color='black'),
                boxprops=dict(linewidth=1.5, color='black')
            )

            for i, patch in enumerate(bp['boxes']):
                label = x_labels[i]
                sign = 'positive' if 'positive' in label else 'negative'
                color = SIGN_COLORS.get(sign, '#cccccc')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)

            for i, data in enumerate(plot_data):
                y = data
                x = np.random.normal(i + 1, 0.05, size=len(y))
                ax_models.scatter(
                    x, y, alpha=0.35, s=25,
                    color='black', edgecolors='gray', linewidth=0.5
                )

            ax_models.set_ylabel('Value', fontsize=13, fontweight='bold')
            ax_models.set_title(
                f'{col} per model (positive vs negative influence)',
                fontsize=13, fontweight='bold', pad=15
            )
            ax_models.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax_models.set_axisbelow(True)
            ax_models.spines['top'].set_visible(False)
            ax_models.spines['right'].set_visible(False)
            ax_models.spines['left'].set_linewidth(1.8)
            ax_models.spines['bottom'].set_linewidth(1.8)
            for tick in ax_models.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha('right')

            global_plot_data = []
            global_labels = []
            for sign in signs:
                data = df.loc[df['influence_sign'] == sign, col].dropna()
                if len(data) > 0:
                    global_plot_data.append(data.values)
                    global_labels.append(sign)

            if global_plot_data:
                bp2 = ax_global.boxplot(
                    global_plot_data,
                    tick_labels=global_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanline=False,
                    notch=False,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(
                        marker='D', markerfacecolor='red',
                        markersize=7, markeredgecolor='darkred',
                        markeredgewidth=1.5
                    ),
                    medianprops=dict(color='darkblue', linewidth=2),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    boxprops=dict(linewidth=1.5, color='black')
                )

                for i, patch in enumerate(bp2['boxes']):
                    sign = global_labels[i]
                    color = SIGN_COLORS.get(sign, '#cccccc')
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)

                for i, data in enumerate(global_plot_data):
                    y = data
                    x = np.random.normal(i + 1, 0.05, size=len(y))
                    ax_global.scatter(
                        x, y, alpha=0.35, s=25,
                        color='black', edgecolors='gray', linewidth=0.5
                    )

                ax_global.set_ylabel('Value', fontsize=13, fontweight='bold')
                ax_global.set_title(
                    f'{col} (all models, positive vs negative influence)',
                    fontsize=13, fontweight='bold', pad=15
                )
                ax_global.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax_global.set_axisbelow(True)
                ax_global.spines['top'].set_visible(False)
                ax_global.spines['right'].set_visible(False)
                ax_global.spines['left'].set_linewidth(1.8)
                ax_global.spines['bottom'].set_linewidth(1.8)
            else:
                ax_global.text(
                    0.5, 0.5,
                    'No data for positive / negative influence',
                    transform=ax_global.transAxes,
                    ha='center', va='center', fontsize=12, color='red'
                )
                ax_global.axis('off')

            fig.suptitle(
                f'Feature Analysis (influence split): '
                f'{feature_base.replace("_", " ").title()}',
                fontsize=16, fontweight='bold', y=0.97
            )

            plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

            stats_text = format_influence_statistics_box(x_labels, plot_data)
            add_bottom_stats_panel(fig, ax_models, stats_text,
                                   width_frac=0.45, y_margin=0.04)

            if global_plot_data:
                global_stats_text = format_influence_statistics_box(
                    global_labels, global_plot_data
                )
                add_bottom_stats_panel(fig, ax_global, global_stats_text,
                                       width_frac=0.30, y_margin=0.04)

            output_file = feature_folder / f'{feature_base}_influence_boxplots.png'
            plt.savefig(
                output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none'
            )
            plt.close(fig)

            print(f"  ✓ Saved: {output_file}")

        else:
            stat_order = ['min', 'mean', 'std', 'max']
            columns_sorted = sorted(
                columns_list,
                key=lambda x: next(
                    (i for i, stat in enumerate(stat_order) if stat == x[1]),
                    999
                )
            )

            for col, stat in columns_sorted:
                stat_label = stat.upper() if stat != 'single' else col
                print(f"    -> Stat: {stat_label} ({col})")

                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                ax_models, ax_global = axes

                plot_data = []
                x_labels = []

                for model in models:
                    for sign in signs:
                        mask = (
                            (df['model'] == model) &
                            (df['influence_sign'] == sign)
                        )
                        data = df.loc[mask, col].dropna()
                        if len(data) > 0:
                            plot_data.append(data.values)
                            x_labels.append(f'{model}\n{sign}')
                        else:
                            plot_data.append(None)
                            x_labels.append(f'{model}\n{sign}')

                non_empty_indices = [
                    i for i, d in enumerate(plot_data)
                    if d is not None and len(d) > 0
                ]
                if not non_empty_indices:
                    print(f"      ⚠️ SKIPPED: No valid data for {col}")
                    plt.close(fig)
                    continue

                plot_data = [plot_data[i] for i in non_empty_indices]
                x_labels = [x_labels[i] for i in non_empty_indices]

                bp = ax_models.boxplot(
                    plot_data,
                    tick_labels=x_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanline=False,
                    notch=False,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(
                        marker='D', markerfacecolor='red',
                        markersize=7, markeredgecolor='darkred',
                        markeredgewidth=1.5
                    ),
                    medianprops=dict(color='darkblue', linewidth=2),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    boxprops=dict(linewidth=1.5, color='black')
                )

                for i, patch in enumerate(bp['boxes']):
                    label = x_labels[i]
                    sign = 'positive' if 'positive' in label else 'negative'
                    color = SIGN_COLORS.get(sign, '#cccccc')
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)

                for i, data in enumerate(plot_data):
                    y = data
                    x = np.random.normal(i + 1, 0.05, size=len(y))
                    ax_models.scatter(
                        x, y, alpha=0.35, s=25,
                        color='black', edgecolors='gray', linewidth=0.5
                    )

                ax_models.set_ylabel('Value', fontsize=13, fontweight='bold')
                ax_models.set_title(
                    f'{feature_base} – {stat_label} per model\n'
                    f'(positive vs negative influence)',
                    fontsize=13, fontweight='bold', pad=15
                )
                ax_models.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax_models.set_axisbelow(True)
                ax_models.spines['top'].set_visible(False)
                ax_models.spines['right'].set_visible(False)
                ax_models.spines['left'].set_linewidth(1.8)
                ax_models.spines['bottom'].set_linewidth(1.8)
                for tick in ax_models.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')

                global_plot_data = []
                global_labels = []
                for sign in signs:
                    data = df.loc[df['influence_sign'] == sign, col].dropna()
                    if len(data) > 0:
                        global_plot_data.append(data.values)
                        global_labels.append(sign)

                if global_plot_data:
                    bp2 = ax_global.boxplot(
                        global_plot_data,
                        tick_labels=global_labels,
                        patch_artist=True,
                        widths=0.6,
                        showmeans=True,
                        meanline=False,
                        notch=False,
                        vert=True,
                        whis=1.5,
                        meanprops=dict(
                            marker='D', markerfacecolor='red',
                            markersize=7, markeredgecolor='darkred',
                            markeredgewidth=1.5
                        ),
                        medianprops=dict(color='darkblue', linewidth=2),
                        whiskerprops=dict(linewidth=1.5, color='black'),
                        capprops=dict(linewidth=1.5, color='black'),
                        boxprops=dict(linewidth=1.5, color='black')
                    )

                    for i, patch in enumerate(bp2['boxes']):
                        sign = global_labels[i]
                        color = SIGN_COLORS.get(sign, '#cccccc')
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(2)

                    for i, data in enumerate(global_plot_data):
                        y = data
                        x = np.random.normal(i + 1, 0.05, size=len(y))
                        ax_global.scatter(
                            x, y, alpha=0.35, s=25,
                            color='black', edgecolors='gray', linewidth=0.5
                        )

                    ax_global.set_ylabel('Value', fontsize=13, fontweight='bold')
                    ax_global.set_title(
                        f'{feature_base} – {stat_label}\n'
                        f'(all models, positive vs negative influence)',
                        fontsize=13, fontweight='bold', pad=15
                    )
                    ax_global.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                    ax_global.set_axisbelow(True)
                    ax_global.spines['top'].set_visible(False)
                    ax_global.spines['right'].set_visible(False)
                    ax_global.spines['left'].set_linewidth(1.8)
                    ax_global.spines['bottom'].set_linewidth(1.8)
                else:
                    ax_global.text(
                        0.5, 0.5,
                        f'No data for {stat_label}',
                        transform=ax_global.transAxes,
                        ha='center', va='center', fontsize=12, color='red'
                    )
                    ax_global.axis('off')

                fig.suptitle(
                    f'Feature Analysis (influence split): '
                    f'{feature_base.replace("_", " ").title()} – {stat_label}',
                    fontsize=16, fontweight='bold', y=0.97
                )

                plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

                stats_text = format_influence_statistics_box(x_labels, plot_data)
                add_bottom_stats_panel(fig, ax_models, stats_text,
                                       width_frac=0.45, y_margin=0.04)

                if global_plot_data:
                    global_stats_text = format_influence_statistics_box(
                        global_labels, global_plot_data
                    )
                    add_bottom_stats_panel(fig, ax_global, global_stats_text,
                                           width_frac=0.30, y_margin=0.04)

                output_file = feature_folder / f'{feature_base}_{stat}_influence_boxplots.png'
                plt.savefig(
                    output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none'
                )
                plt.close(fig)

                print(f"      ✓ Saved: {output_file}")

    print(f"\n{'='*80}")
    print(f"✅ Influence-split boxplot visualizations saved to: {base_folder}")
    print(f"✅ Ready for academic thesis presentation!")
    print(f"{'='*80}\n")

def viz_best_vs_worst_boxplots(
        features_df,
        base_output_folder=Path('./'),
        group_col='group',
        importance_col='abs_importance'
):
    setup_professional_style()

    base_folder = Path(f'{base_output_folder}/visualizations_boxplot_best_worst')
    base_folder.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*80}")
    print("Creating BEST vs WORST boxplot visualizations for academic thesis...")
    print(f"{'='*80}\n")

    if group_col not in features_df.columns:
        raise ValueError(f"Column '{group_col}' not found in features_df")

    df = features_df.copy()
    df = df[df[group_col].isin(['best', 'worst'])].copy()
    if df.empty:
        print("⚠️ No 'best' / 'worst' samples found.")
        return

    models = sorted(df['model'].dropna().unique())
    groups = ['worst', 'best']

    exclude_cols = {
        'model', 'track', 'patch_key', 'data_type',
        group_col, importance_col, 'influence_sign'
    }
    occlusion_meta_cols = {
        'group', 'rank', 'importance', 'abs_importance',
        'tstart', 'tend', 'fstart', 'fend',
        'start_time_sec', 'end_time_sec',
        'patch_type', 'model', 'track_stem'
    }

    all_cols = [
        col for col in df.columns
        if col not in exclude_cols and col not in occlusion_meta_cols
    ]

    feature_groups = defaultdict(list)

    for col in all_cols:
        parts = col.split('_')
        if len(parts) > 1 and parts[-1] in ['min', 'mean', 'std', 'max']:
            base_name = '_'.join(parts[:-1])
            stat = parts[-1]
        else:
            base_name = col
            stat = 'single'
        feature_groups[base_name].append((col, stat))

    print(f"Found {len(feature_groups)} feature groups\n")

    for feature_base, columns_list in sorted(feature_groups.items()):
        print(f"Processing feature: {feature_base}")

        feature_folder = base_folder / feature_base
        feature_folder.mkdir(exist_ok=True, parents=True)

        if len(columns_list) == 1 and columns_list[0][1] == 'single':
            col = columns_list[0][0]

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            ax_models, ax_global = axes

            plot_data = []
            x_labels = []

            for model in models:
                for g in groups:
                    mask = (df['model'] == model) & (df[group_col] == g)
                    data = df.loc[mask, col].dropna()
                    if len(data) > 0:
                        plot_data.append(data.values)
                        x_labels.append(f'{model}\n{g}')
                    else:
                        plot_data.append(None)
                        x_labels.append(f'{model}\n{g}')

            non_empty_indices = [
                i for i, d in enumerate(plot_data)
                if d is not None and len(d) > 0
            ]
            if not non_empty_indices:
                print(f"  ⚠️ SKIPPED: No valid data for feature '{col}'")
                plt.close(fig)
                continue

            plot_data = [plot_data[i] for i in non_empty_indices]
            x_labels = [x_labels[i] for i in non_empty_indices]

            bp = ax_models.boxplot(
                plot_data,
                tick_labels=x_labels,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                meanline=False,
                notch=False,
                vert=True,
                whis=1.5,
                meanprops=dict(
                    marker='D', markerfacecolor='red',
                    markersize=7, markeredgecolor='darkred',
                    markeredgewidth=1.5
                ),
                medianprops=dict(color='darkblue', linewidth=2),
                whiskerprops=dict(linewidth=1.5, color='black'),
                capprops=dict(linewidth=1.5, color='black'),
                boxprops=dict(linewidth=1.5, color='black')
            )

            for i, patch in enumerate(bp['boxes']):
                label = x_labels[i]
                g = 'best' if 'best' in label else 'worst'
                color = '#2ca02c' if g == 'best' else '#d62728'
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)

            for i, data in enumerate(plot_data):
                y = data
                x = np.random.normal(i + 1, 0.05, size=len(y))
                ax_models.scatter(
                    x, y, alpha=0.35, s=25,
                    color='black', edgecolors='gray', linewidth=0.5
                )

            ax_models.set_ylabel('Value', fontsize=13, fontweight='bold')
            ax_models.set_title(
                f'{col} per model (BEST vs WORST)',
                fontsize=13, fontweight='bold', pad=15
            )
            ax_models.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax_models.set_axisbelow(True)
            ax_models.spines['top'].set_visible(False)
            ax_models.spines['right'].set_visible(False)
            ax_models.spines['left'].set_linewidth(1.8)
            ax_models.spines['bottom'].set_linewidth(1.8)
            for tick in ax_models.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha('right')

            global_plot_data = []
            global_labels = []
            for g in groups:
                data = df.loc[df[group_col] == g, col].dropna()
                if len(data) > 0:
                    global_plot_data.append(data.values)
                    global_labels.append(g)

            if global_plot_data:
                bp2 = ax_global.boxplot(
                    global_plot_data,
                    tick_labels=global_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanline=False,
                    notch=False,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(
                        marker='D', markerfacecolor='red',
                        markersize=7, markeredgecolor='darkred',
                        markeredgewidth=1.5
                    ),
                    medianprops=dict(color='darkblue', linewidth=2),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    boxprops=dict(linewidth=1.5, color='black')
                )

                for i, patch in enumerate(bp2['boxes']):
                    g = global_labels[i]
                    color = '#2ca02c' if g == 'best' else '#d62728'
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)

                for i, data in enumerate(global_plot_data):
                    y = data
                    x = np.random.normal(i + 1, 0.05, size=len(y))
                    ax_global.scatter(
                        x, y, alpha=0.35, s=25,
                        color='black', edgecolors='gray', linewidth=0.5
                    )

                ax_global.set_ylabel('Value', fontsize=13, fontweight='bold')
                ax_global.set_title(
                    f'{col} (all models, BEST vs WORST)',
                    fontsize=13, fontweight='bold', pad=15
                )
                ax_global.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax_global.set_axisbelow(True)
                ax_global.spines['top'].set_visible(False)
                ax_global.spines['right'].set_visible(False)
                ax_global.spines['left'].set_linewidth(1.8)
                ax_global.spines['bottom'].set_linewidth(1.8)
            else:
                ax_global.text(
                    0.5, 0.5,
                    'No data for BEST / WORST',
                    transform=ax_global.transAxes,
                    ha='center', va='center', fontsize=12, color='red'
                )
                ax_global.axis('off')

            fig.suptitle(
                f'Feature Analysis (BEST vs WORST): '
                f'{feature_base.replace("_", " ").title()}',
                fontsize=16, fontweight='bold', y=0.97
            )

            plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

            stats_text = format_influence_statistics_box(x_labels, plot_data)
            add_bottom_stats_panel(fig, ax_models, stats_text,
                                   width_frac=0.45, y_margin=0.04)

            if global_plot_data:
                global_stats_text = format_influence_statistics_box(
                    global_labels, global_plot_data
                )
                add_bottom_stats_panel(fig, ax_global, global_stats_text,
                                       width_frac=0.30, y_margin=0.04)

            output_file = feature_folder / f'{feature_base}_best_vs_worst.png'
            plt.savefig(
                output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none'
            )
            plt.close(fig)

            print(f"  ✓ Saved: {output_file}")

        else:
            stat_order = ['min', 'mean', 'std', 'max']
            columns_sorted = sorted(
                columns_list,
                key=lambda x: next(
                    (i for i, stat in enumerate(stat_order) if stat == x[1]),
                    999
                )
            )

            for col, stat in columns_sorted:
                stat_label = stat.upper() if stat != 'single' else col
                print(f"    -> Stat: {stat_label} ({col})")

                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                ax_models, ax_global = axes

                plot_data = []
                x_labels = []

                for model in models:
                    for g in groups:
                        mask = (df['model'] == model) & (df[group_col] == g)
                        data = df.loc[mask, col].dropna()
                        if len(data) > 0:
                            plot_data.append(data.values)
                            x_labels.append(f'{model}\n{g}')
                        else:
                            plot_data.append(None)
                            x_labels.append(f'{model}\n{g}')

                non_empty_indices = [
                    i for i, d in enumerate(plot_data)
                    if d is not None and len(d) > 0
                ]
                if not non_empty_indices:
                    print(f"      ⚠️ SKIPPED: No valid data for {col}")
                    plt.close(fig)
                    continue

                plot_data = [plot_data[i] for i in non_empty_indices]
                x_labels = [x_labels[i] for i in non_empty_indices]

                bp = ax_models.boxplot(
                    plot_data,
                    tick_labels=x_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanline=False,
                    notch=False,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(
                        marker='D', markerfacecolor='red',
                        markersize=7, markeredgecolor='darkred',
                        markeredgewidth=1.5
                    ),
                    medianprops=dict(color='darkblue', linewidth=2),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    boxprops=dict(linewidth=1.5, color='black')
                )

                for i, patch in enumerate(bp['boxes']):
                    label = x_labels[i]
                    g = 'best' if 'best' in label else 'worst'
                    color = '#2ca02c' if g == 'best' else '#d62728'
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)

                for i, data in enumerate(plot_data):
                    y = data
                    x = np.random.normal(i + 1, 0.05, size=len(y))
                    ax_models.scatter(
                        x, y, alpha=0.35, s=25,
                        color='black', edgecolors='gray', linewidth=0.5
                    )

                ax_models.set_ylabel('Value', fontsize=13, fontweight='bold')
                ax_models.set_title(
                    f'{feature_base} – {stat_label} per model\n'
                    f'(BEST vs WORST)',
                    fontsize=13, fontweight='bold', pad=15
                )
                ax_models.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax_models.set_axisbelow(True)
                ax_models.spines['top'].set_visible(False)
                ax_models.spines['right'].set_visible(False)
                ax_models.spines['left'].set_linewidth(1.8)
                ax_models.spines['bottom'].set_linewidth(1.8)
                for tick in ax_models.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')

                global_plot_data = []
                global_labels = []
                for g in groups:
                    data = df.loc[df[group_col] == g, col].dropna()
                    if len(data) > 0:
                        global_plot_data.append(data.values)
                        global_labels.append(g)

                if global_plot_data:
                    bp2 = ax_global.boxplot(
                        global_plot_data,
                        tick_labels=global_labels,
                        patch_artist=True,
                        widths=0.6,
                        showmeans=True,
                        meanline=False,
                        notch=False,
                        vert=True,
                        whis=1.5,
                        meanprops=dict(
                            marker='D', markerfacecolor='red',
                            markersize=7, markeredgecolor='darkred',
                            markeredgewidth=1.5
                        ),
                        medianprops=dict(color='darkblue', linewidth=2),
                        whiskerprops=dict(linewidth=1.5, color='black'),
                        capprops=dict(linewidth=1.5, color='black'),
                        boxprops=dict(linewidth=1.5, color='black')
                    )

                    for i, patch in enumerate(bp2['boxes']):
                        g = global_labels[i]
                        color = '#2ca02c' if g == 'best' else '#d62728'
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(2)

                    for i, data in enumerate(global_plot_data):
                        y = data
                        x = np.random.normal(i + 1, 0.05, size=len(y))
                        ax_global.scatter(
                            x, y, alpha=0.35, s=25,
                            color='black', edgecolors='gray', linewidth=0.5
                        )

                    ax_global.set_ylabel('Value', fontsize=13, fontweight='bold')
                    ax_global.set_title(
                        f'{feature_base} – {stat_label}\n'
                        f'(all models, BEST vs WORST)',
                        fontsize=13, fontweight='bold', pad=15
                    )
                    ax_global.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                    ax_global.set_axisbelow(True)
                    ax_global.spines['top'].set_visible(False)
                    ax_global.spines['right'].set_visible(False)
                    ax_global.spines['left'].set_linewidth(1.8)
                    ax_global.spines['bottom'].set_linewidth(1.8)
                else:
                    ax_global.text(
                        0.5, 0.5,
                        f'No data for {stat_label}',
                        transform=ax_global.transAxes,
                        ha='center', va='center', fontsize=12, color='red'
                    )
                    ax_global.axis('off')

                fig.suptitle(
                    f'Feature Analysis (BEST vs WORST): '
                    f'{feature_base.replace("_", " ").title()} – {stat_label}',
                    fontsize=16, fontweight='bold', y=0.97
                )

                plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

                stats_text = format_influence_statistics_box(x_labels, plot_data)
                add_bottom_stats_panel(fig, ax_models, stats_text,
                                       width_frac=0.45, y_margin=0.04)

                if global_plot_data:
                    global_stats_text = format_influence_statistics_box(
                        global_labels, global_plot_data
                    )
                    add_bottom_stats_panel(fig, ax_global, global_stats_text,
                                           width_frac=0.30, y_margin=0.04)

                output_file = feature_folder / f'{feature_base}_{stat}_best_vs_worst.png'
                plt.savefig(
                    output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none'
                )
                plt.close(fig)

                print(f"      ✓ Saved: {output_file}")

    print(f"\n{'='*80}")
    print(f"✅ BEST vs WORST boxplot visualizations saved to: {base_folder}")
    print(f"✅ Ready for academic thesis presentation!")
    print(f"{'='*80}\n")

def viz_most_influential_pos_neg_boxplots(
        features_df,
        base_output_folder=Path('./'),
        group_col='group',
        importance_col='importance',
        sign_col='influence_sign'
):
    setup_professional_style()

    base_folder = Path(f'{base_output_folder}/visualizations_boxplot_most_influential')
    base_folder.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*80}")
    print("Creating MOST-INFLUENTIAL (positive vs negative) boxplot visualizations...")
    print(f"{'='*80}\n")

    if group_col not in features_df.columns:
        raise ValueError(f"Column '{group_col}' not found in features_df")

    df = features_df.copy()
    df = df[df[group_col] == 'most_influential'].copy()
    if df.empty:
        print("⚠️ No 'most_influential' samples found.")
        return

    if sign_col not in df.columns:
        if importance_col not in df.columns:
            raise ValueError(
                f"Neither sign_col '{sign_col}' nor importance_col "
                f"'{importance_col}' found in DataFrame."
            )
        df['influence_sign'] = np.where(df[importance_col] >= 0,
                                        'positive', 'negative')
        sign_col = 'influence_sign'

    models = sorted(df['model'].dropna().unique())
    signs = ['negative', 'positive']

    exclude_cols = {
        'model', 'track', 'patch_key', 'data_type',
        group_col, importance_col, sign_col
    }
    occlusion_meta_cols = {
        'group', 'rank', 'importance', 'abs_importance',
        'tstart', 'tend', 'fstart', 'fend',
        'start_time_sec', 'end_time_sec',
        'patch_type', 'model', 'track_stem'
    }

    all_cols = [
        col for col in df.columns
        if col not in exclude_cols and col not in occlusion_meta_cols
    ]

    feature_groups = defaultdict(list)
    for col in all_cols:
        parts = col.split('_')
        if len(parts) > 1 and parts[-1] in ['min', 'mean', 'std', 'max']:
            base_name = '_'.join(parts[:-1])
            stat = parts[-1]
        else:
            base_name = col
            stat = 'single'
        feature_groups[base_name].append((col, stat))

    print(f"Found {len(feature_groups)} feature groups for most_influential\n")

    for feature_base, columns_list in sorted(feature_groups.items()):
        print(f"Processing feature: {feature_base}")

        feature_folder = base_folder / feature_base
        feature_folder.mkdir(exist_ok=True, parents=True)

        if len(columns_list) == 1 and columns_list[0][1] == 'single':
            col = columns_list[0][0]

            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            ax_models, ax_global = axes

            plot_data = []
            x_labels = []

            for model in models:
                for s in signs:
                    mask = (df['model'] == model) & (df[sign_col] == s)
                    data = df.loc[mask, col].dropna()
                    if len(data) > 0:
                        plot_data.append(data.values)
                        x_labels.append(f'{model}\n{s}')
                    else:
                        plot_data.append(None)
                        x_labels.append(f'{model}\n{s}')

            non_empty_indices = [
                i for i, d in enumerate(plot_data)
                if d is not None and len(d) > 0
            ]
            if not non_empty_indices:
                print(f"  ⚠️ SKIPPED: No valid data for feature '{col}'")
                plt.close(fig)
                continue

            plot_data = [plot_data[i] for i in non_empty_indices]
            x_labels = [x_labels[i] for i in non_empty_indices]

            bp = ax_models.boxplot(
                plot_data,
                tick_labels=x_labels,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                meanline=False,
                notch=False,
                vert=True,
                whis=1.5,
                meanprops=dict(
                    marker='D', markerfacecolor='red',
                    markersize=7, markeredgecolor='darkred',
                    markeredgewidth=1.5
                ),
                medianprops=dict(color='darkblue', linewidth=2),
                whiskerprops=dict(linewidth=1.5, color='black'),
                capprops=dict(linewidth=1.5, color='black'),
                boxprops=dict(linewidth=1.5, color='black')
            )

            for i, patch in enumerate(bp['boxes']):
                label = x_labels[i]
                s = 'positive' if 'positive' in label else 'negative'
                color = SIGN_COLORS.get(s, '#cccccc')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)

            for i, data in enumerate(plot_data):
                y = data
                x = np.random.normal(i + 1, 0.05, size=len(y))
                ax_models.scatter(
                    x, y, alpha=0.35, s=25,
                    color='black', edgecolors='gray', linewidth=0.5
                )

            ax_models.set_ylabel('Value', fontsize=13, fontweight='bold')
            ax_models.set_title(
                f'{col} per model (most_influential, positive vs negative)',
                fontsize=13, fontweight='bold', pad=15
            )
            ax_models.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
            ax_models.set_axisbelow(True)
            ax_models.spines['top'].set_visible(False)
            ax_models.spines['right'].set_visible(False)
            ax_models.spines['left'].set_linewidth(1.8)
            ax_models.spines['bottom'].set_linewidth(1.8)
            for tick in ax_models.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha('right')

            global_plot_data = []
            global_labels = []
            for s in signs:
                data = df.loc[df[sign_col] == s, col].dropna()
                if len(data) > 0:
                    global_plot_data.append(data.values)
                    global_labels.append(s)

            if global_plot_data:
                bp2 = ax_global.boxplot(
                    global_plot_data,
                    tick_labels=global_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanline=False,
                    notch=False,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(
                        marker='D', markerfacecolor='red',
                        markersize=7, markeredgecolor='darkred',
                        markeredgewidth=1.5
                    ),
                    medianprops=dict(color='darkblue', linewidth=2),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    boxprops=dict(linewidth=1.5, color='black')
                )

                for i, patch in enumerate(bp2['boxes']):
                    s = global_labels[i]
                    color = SIGN_COLORS.get(s, '#cccccc')
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)

                for i, data in enumerate(global_plot_data):
                    y = data
                    x = np.random.normal(i + 1, 0.05, size=len(y))
                    ax_global.scatter(
                        x, y, alpha=0.35, s=25,
                        color='black', edgecolors='gray', linewidth=0.5
                    )

                ax_global.set_ylabel('Value', fontsize=13, fontweight='bold')
                ax_global.set_title(
                    f'{col} (most_influential, all models, positive vs negative)',
                    fontsize=13, fontweight='bold', pad=15
                )
                ax_global.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax_global.set_axisbelow(True)
                ax_global.spines['top'].set_visible(False)
                ax_global.spines['right'].set_visible(False)
                ax_global.spines['left'].set_linewidth(1.8)
                ax_global.spines['bottom'].set_linewidth(1.8)
            else:
                ax_global.text(
                    0.5, 0.5,
                    'No data for positive / negative (most_influential)',
                    transform=ax_global.transAxes,
                    ha='center', va='center', fontsize=12, color='red'
                )
                ax_global.axis('off')

            fig.suptitle(
                f'Feature Analysis (most_influential, influence split): '
                f'{feature_base.replace("_", " ").title()}',
                fontsize=16, fontweight='bold', y=0.97
            )

            plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

            stats_text = format_influence_statistics_box(x_labels, plot_data)
            add_bottom_stats_panel(fig, ax_models, stats_text,
                                   width_frac=0.45, y_margin=0.04)

            if global_plot_data:
                global_stats_text = format_influence_statistics_box(
                    global_labels, global_plot_data
                )
                add_bottom_stats_panel(fig, ax_global, global_stats_text,
                                       width_frac=0.30, y_margin=0.04)

            output_file = feature_folder / f'{feature_base}_most_influential_pos_neg.png'
            plt.savefig(
                output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none'
            )
            plt.close(fig)

            print(f"  ✓ Saved: {output_file}")

        else:
            stat_order = ['min', 'mean', 'std', 'max']
            columns_sorted = sorted(
                columns_list,
                key=lambda x: next(
                    (i for i, stat in enumerate(stat_order) if stat == x[1]),
                    999
                )
            )

            for col, stat in columns_sorted:
                stat_label = stat.upper() if stat != 'single' else col
                print(f"    -> Stat: {stat_label} ({col})")

                fig, axes = plt.subplots(1, 2, figsize=(20, 8))
                ax_models, ax_global = axes

                plot_data = []
                x_labels = []

                for model in models:
                    for s in signs:
                        mask = (df['model'] == model) & (df[sign_col] == s)
                        data = df.loc[mask, col].dropna()
                        if len(data) > 0:
                            plot_data.append(data.values)
                            x_labels.append(f'{model}\n{s}')
                        else:
                            plot_data.append(None)
                            x_labels.append(f'{model}\n{s}')

                non_empty_indices = [
                    i for i, d in enumerate(plot_data)
                    if d is not None and len(d) > 0
                ]
                if not non_empty_indices:
                    print(f"      ⚠️ SKIPPED: No valid data for {col}")
                    plt.close(fig)
                    continue

                plot_data = [plot_data[i] for i in non_empty_indices]
                x_labels = [x_labels[i] for i in non_empty_indices]

                bp = ax_models.boxplot(
                    plot_data,
                    tick_labels=x_labels,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanline=False,
                    notch=False,
                    vert=True,
                    whis=1.5,
                    meanprops=dict(
                        marker='D', markerfacecolor='red',
                        markersize=7, markeredgecolor='darkred',
                        markeredgewidth=1.5
                    ),
                    medianprops=dict(color='darkblue', linewidth=2),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    boxprops=dict(linewidth=1.5, color='black')
                )

                for i, patch in enumerate(bp['boxes']):
                    label = x_labels[i]
                    s = 'positive' if 'positive' in label else 'negative'
                    color = SIGN_COLORS.get(s, '#cccccc')
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)

                for i, data in enumerate(plot_data):
                    y = data
                    x = np.random.normal(i + 1, 0.05, size=len(y))
                    ax_models.scatter(
                        x, y, alpha=0.35, s=25,
                        color='black', edgecolors='gray', linewidth=0.5
                    )

                ax_models.set_ylabel('Value', fontsize=13, fontweight='bold')
                ax_models.set_title(
                    f'{feature_base} – {stat_label} per model\n'
                    f'(most_influential, positive vs negative)',
                    fontsize=13, fontweight='bold', pad=15
                )
                ax_models.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax_models.set_axisbelow(True)
                ax_models.spines['top'].set_visible(False)
                ax_models.spines['right'].set_visible(False)
                ax_models.spines['left'].set_linewidth(1.8)
                ax_models.spines['bottom'].set_linewidth(1.8)
                for tick in ax_models.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')

                global_plot_data = []
                global_labels = []
                for s in signs:
                    data = df.loc[df[sign_col] == s, col].dropna()
                    if len(data) > 0:
                        global_plot_data.append(data.values)
                        global_labels.append(s)

                if global_plot_data:
                    bp2 = ax_global.boxplot(
                        global_plot_data,
                        tick_labels=global_labels,
                        patch_artist=True,
                        widths=0.6,
                        showmeans=True,
                        meanline=False,
                        notch=False,
                        vert=True,
                        whis=1.5,
                        meanprops=dict(
                            marker='D', markerfacecolor='red',
                            markersize=7, markeredgecolor='darkred',
                            markeredgewidth=1.5
                        ),
                        medianprops=dict(color='darkblue', linewidth=2),
                        whiskerprops=dict(linewidth=1.5, color='black'),
                        capprops=dict(linewidth=1.5, color='black'),
                        boxprops=dict(linewidth=1.5, color='black')
                    )

                    for i, patch in enumerate(bp2['boxes']):
                        s = global_labels[i]
                        color = SIGN_COLORS.get(s, '#cccccc')
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(2)

                    for i, data in enumerate(global_plot_data):
                        y = data
                        x = np.random.normal(i + 1, 0.05, size=len(y))
                        ax_global.scatter(
                            x, y, alpha=0.35, s=25,
                            color='black', edgecolors='gray', linewidth=0.5
                        )

                    ax_global.set_ylabel('Value', fontsize=13, fontweight='bold')
                    ax_global.set_title(
                        f'{feature_base} – {stat_label}\n'
                        f'(most_influential, all models, positive vs negative)',
                        fontsize=13, fontweight='bold', pad=15
                    )
                    ax_global.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                    ax_global.set_axisbelow(True)
                    ax_global.spines['top'].set_visible(False)
                    ax_global.spines['right'].set_visible(False)
                    ax_global.spines['left'].set_linewidth(1.8)
                    ax_global.spines['bottom'].set_linewidth(1.8)
                else:
                    ax_global.text(
                        0.5, 0.5,
                        f'No data for {stat_label} (most_influential)',
                        transform=ax_global.transAxes,
                        ha='center', va='center', fontsize=12, color='red'
                    )
                    ax_global.axis('off')

                fig.suptitle(
                    f'Feature Analysis (most_influential, influence split): '
                    f'{feature_base.replace("_", " ").title()} – {stat_label}',
                    fontsize=16, fontweight='bold', y=0.97
                )

                plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

                stats_text = format_influence_statistics_box(x_labels, plot_data)
                add_bottom_stats_panel(fig, ax_models, stats_text,
                                       width_frac=0.45, y_margin=0.04)

                if global_plot_data:
                    global_stats_text = format_influence_statistics_box(
                        global_labels, global_plot_data
                    )
                    add_bottom_stats_panel(fig, ax_global, global_stats_text,
                                           width_frac=0.30, y_margin=0.04)

                output_file = feature_folder / f'{feature_base}_{stat}_most_influential_pos_neg.png'
                plt.savefig(
                    output_file, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none'
                )
                plt.close(fig)

                print(f"      ✓ Saved: {output_file}")

    print(f"\n{'='*80}")
    print(f"✅ MOST-INFLUENTIAL (positive vs negative) boxplots saved to: {base_folder}")
    print(f"✅ Ready for academic thesis presentation!")
    print(f"{'='*80}\n")

def viz_single_feature_vs_importance_in_group(
    group_df,
    feature_col,
    feature_folder,
    feature_label,
    importance_col="importance",
    sign_col="influence_sign",
):
    required_cols = [feature_col, importance_col, "model", sign_col]
    for col in required_cols:
        if col not in group_df.columns:
            return

    feature_df = group_df[required_cols].dropna(
        subset=[feature_col, importance_col]
    ).copy()
    if feature_df.empty:
        return

    models = sorted(feature_df["model"].dropna().unique().tolist())
    if not models:
        return

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(4 * n_models, 6), sharey=True
    )
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_df = feature_df[feature_df["model"] == model_name]
        if model_df.empty:
            ax.set_visible(False)
            continue

        color_hex = PROFESSIONAL_COLORS.get(model_name, "333333")
        if not color_hex.startswith("#"):
            color_hex = "#" + color_hex

        positive_df = model_df[model_df[sign_col] == "positive"]
        negative_df = model_df[model_df[sign_col] == "negative"]

        if not positive_df.empty:
            ax.scatter(
                positive_df[feature_col],
                positive_df[importance_col],
                color=color_hex,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
                s=50,
                marker="o",
                label="positive",
            )

        if not negative_df.empty:
            ax.scatter(
                negative_df[feature_col],
                negative_df[importance_col],
                color=color_hex,
                alpha=0.4,
                edgecolors="black",
                linewidth=1.8,
                s=70,
                marker="X",
                label="negative",
            )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_title(
            model_name,
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        x_vals = model_df[feature_col].values
        y_vals = model_df[importance_col].values
        stats_text = f"n = {len(model_df)}"
        if len(model_df) >= 2:
            try:
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
            except Exception:
                corr = np.nan
            if not np.isnan(corr):
                stats_text += f"\nPearson r = {corr:.3f}"

        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.9,
                edgecolor="black",
                linewidth=0.8,
            ),
        )

    fig.supxlabel(feature_label, fontsize=13, fontweight="bold")
    fig.supylabel("Patch importance", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=45, labelsize=10)
        ax.xaxis.set_tick_params(labelbottom=True)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            label="positive",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markeredgecolor="black",
            label="negative",
            markersize=8,
            linewidth=1.8,
        ),
    ]
    fig.legend(
        handles=legend_elements,
        title="Signs of influence",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=9,
        title_fontsize=10,
    )

    fig.suptitle(
        f"{feature_label} vs importance – per model",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.93))

    safe_label = feature_label.replace(" ", "_").replace("/", "_")
    outfile = feature_folder / f"{safe_label}_vs_importance_per_model.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(outfile)


def viz_feature_values_vs_importance_by_group(
    features_df,
    base_output_folder=Path("."),
    importance_col="importance",
    window_col="window_label",
):
    setup_professional_style()
    df = features_df.copy()

    if importance_col not in df.columns:
        raise ValueError(f"Column '{importance_col}' not found")

    df["influence_sign"] = np.where(
        df[importance_col] >= 0, "positive", "negative"
    )

    if window_col not in df.columns:
        raise ValueError(f"Column '{window_col}' not found")

    windows = sorted(df[window_col].dropna().unique().tolist())

    print(f"Processing {len(windows)} windows for feature vs importance...")

    for window_name in windows:
        window_df = df[df[window_col] == window_name].copy()
        if window_df.empty:
            print(f"Skipping empty window {window_name}")
            continue

        window_safe = str(window_name).replace(" ", "_")
        window_base_folder = (
            Path(base_output_folder)
            / "by_group_feature_vs_importance"
            / window_safe
        )
        window_base_folder.mkdir(parents=True, exist_ok=True)

        exclude_cols = [
            "model",
            "track",
            "patch_key",
            "data_type",
            importance_col,
            "influence_sign",
            window_col,
        ]
        occlusion_meta_cols = [
            "group",
            "rank",
            "importance",
            "abs_importance",
            "t_start",
            "t_end",
            "f_start",
            "f_end",
            "start_time_sec",
            "end_time_sec",
            "patch_type",
            "model",
            "track_stem",
        ]

        all_cols = [
            col
            for col in window_df.columns
            if col not in exclude_cols
            and col not in occlusion_meta_cols
            and window_df[col].dtype in (np.float64, np.float32, np.int64, np.int32)
            and window_df[col].notna().sum() > 0
        ]

        feature_windows = defaultdict(list)
        for col in all_cols:
            parts = col.split("_")
            if len(parts) > 1 and parts[-1] in ("min", "mean", "std", "max"):
                base_name = "_".join(parts[:-1])
                stat = parts[-1]
            else:
                base_name = col
                stat = "single"
            feature_windows[base_name].append((col, stat))

        print(f"{window_name}: {len(feature_windows)} feature windows")

        for feature_base, columns_list in sorted(feature_windows.items()):
            feature_folder = window_base_folder / feature_base
            feature_folder.mkdir(parents=True, exist_ok=True)

            if len(columns_list) == 1 and columns_list[0][1] == "single":
                col, _ = columns_list[0]
                feature_label = f"{feature_base}"
                viz_single_feature_vs_importance_in_group(
                    window_df,
                    col,
                    feature_folder,
                    feature_label,
                    importance_col=importance_col,
                    sign_col="influence_sign",
                )
            else:
                stat_order = ["min", "mean", "std", "max"]
                columns_sorted = sorted(
                    columns_list,
                    key=lambda x: next(
                        (i for i, s in enumerate(stat_order) if s == x[1]), 999
                    ),
                )
                for col, stat in columns_sorted:
                    if stat == "single":
                        feature_label = feature_base
                    else:
                        feature_label = f"{feature_base} ({stat.upper()})"
                    viz_single_feature_vs_importance_in_group(
                        window_df,
                        col,
                        feature_folder,
                        feature_label,
                        importance_col=importance_col,
                        sign_col="influence_sign",
                    )

        print(f"{window_base_folder} done")


def main():
    args = parse_args()
    config = load_yaml(Path(args.config))

    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    data_root = Path(data_cfg.get("features_path"))
    result_root = Path(output_cfg.get("result_path"))

    features_path = data_root / "occlusion_patches_features.json"
    output_root = result_root / "features_visualization"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Output root: {output_root}")
    print("Visualizing occlusion feature importance results")
    print("=" * 70)

    features_df, features_to_analyze = load_and_prepare_data_full(features_path)
    print(f"\n✓ Data loaded: {len(features_df)} samples, {len(features_to_analyze)} features")
    print(f"✓ Models: {features_df['model'].value_counts().to_dict()}\n")

    features_df = add_group_from_patch_key(features_df)
    features_df = add_window_label_from_patch_meta(features_df)

    viz2_real_vs_generated_boxplots_with_influence(
        features_df,
        base_output_folder=output_root,
    )

    viz_best_vs_worst_boxplots(
        features_df,
        base_output_folder=output_root
    )

    viz_most_influential_pos_neg_boxplots(
        features_df,
        base_output_folder=output_root
    )

    viz_feature_values_vs_importance_by_group(
        features_df,
        base_output_folder=output_root,
    )

if __name__ == "__main__":
    main()