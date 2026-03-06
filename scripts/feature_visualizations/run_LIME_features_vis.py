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
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    ap = argparse.ArgumentParser(
        description="Visualize AudioLIME components feature importance results."
    )
    ap.add_argument(
        "--config",
        default=(str(ROOT / "configs" / "AudioLIME_configs" / "lime_features_vis.yaml")),
        help="Path to the YAML configuration file."
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
                            result[col_name] = val
                    
                    return result

def load_and_prepare_data_full(json_file):
    """
    Load JSON data and preserve ALL sub-features from nested structure.
    
    Data structure example:
    {
        model_name: {
            track_id: {
                "type": "full_track" | "segment",
                "components": {
                    component_key: {
                        "features": {
                            "duration": 120.0,
                            "rms_wave": {"min": ..., "mean": ..., "std": ..., "max": ...},
                            "jitter": {"jitter_local": ..., "jitter_rap": ..., ...},
                            ...
                        },
                        "component_meta": {
                            "importance": -0.1234,
                            "model": "ElevenLabs" | "REAL" | "SUNO" | "SUNO_PRO" | "UDIO",
                            "track_stem": "some_track_name"
                            "component_name": "vocals0" | "drums0" | "bass0" | "other0",
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
    - DataFrame with collumns: model, track, band_key, component, component_type, data_type, {all_features}]
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
            
            if not isinstance(track_data, dict) or 'components' not in track_data:
                continue
            
            components_root = track_data.get('components', {})
            # track_type = track_data.get('type', 'unknown')

            for component_key, component_data in components_root.items():

                if not isinstance(component_data, dict) or 'features' not in component_data:
                    continue

                features = component_data.get('features', {})
                component_meta = component_data.get('component_meta', {})
                            
                row = {
                    'model': model_name,
                    'track': track_key,
                    'component_key': component_key,
                    'component_type': "POSITIVE" if component_meta.get('importance', 0) >= 0 else "NEGATIVE",
                    'data_type': type_mapping.get(model_name, model_name),
                    **flatten_feature(component_meta)
                }
                
                flattened = flatten_feature(features)
                row.update(flattened)
                
                all_rows.append(row)
    
    features_df = pd.DataFrame(all_rows)
    
    if features_df.empty:
        print("⚠️ Warning: No data loaded from JSON file!")
        return features_df, []
    
    exclude_cols = {'model', 'track', 'component_key', 'component_type', 'data_type'}
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
    """
    labels: labels on X axis, e.g. ['REAL\\nnegative', 'REAL\\npositive', ...]
    plot_data: list of 1D arrays (as in boxplot)
    """
    rows = []

    header = ["Component", "Mean", "Std", "Count"]
    rows.append(header)

    for label, data in zip(labels, plot_data):
        if data is None or len(data) == 0:
            continue

        if "\n" in label:
            model, sign = label.split("\n", 1)
            component_name = f"{model} ({sign})"
        else:
            component_name = label

        mean_str = f"{np.mean(data):.4f}"
        std_str = f"{np.std(data):.4f}"
        count_str = f"{len(data)}"

        row = [component_name, mean_str, std_str, count_str]
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

def viz_component_pos_neg_boxplots(
    features_df,
    base_output_folder=Path('./'),
    importance_col='importance',
    sign_col='influence_sign'
):
    setup_professional_style()
    
    base_folder = Path(f'{base_output_folder}/visualizations_boxplot_component_pos_neg')
    base_folder.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("Creating COMPONENT-SPLIT boxplot visualizations (positive vs negative)...")
    print(f"{'='*80}\n")
    
    if importance_col not in features_df.columns:
        raise ValueError(f"Column '{importance_col}' not found in features_df")
    
    df = features_df.copy()
    if sign_col not in df.columns:
        if importance_col not in df.columns:
            raise ValueError(
                f"Neither sign_col '{sign_col}' nor importance_col "
                f"'{importance_col}' found in DataFrame."
            )    
        df['influence_sign'] = np.where(df[importance_col] >= 0, 'positive', 'negative')
        sign_col = 'influence_sign'
    
    models = sorted(df['model'].dropna().unique())
    signs = ['positive', 'negative']

    exclude_cols = {
        'model', 'track', 'component_key', 'data_type',
        importance_col, 'influence_sign'
    }
    
    band_meta_cols = {
        'component', 'importance', 'abs_importance',
        'low_freq', 'high_freq', 'band_type', 'model', 'track_stem'
    }
    
    all_cols = [
        col for col in df.columns
        if (
            col not in exclude_cols and 
            col not in band_meta_cols and
            df[col].dtype in ['float64', 'float32', 'int64', 'int32'] and
            df[col].notna().sum() > 0
        )
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
                f'{col} per component (positive vs negative influence)',
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
                    f'{col} (all components, positive vs negative influence)',
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
                f'Feature Analysis (component split): '
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
            
            output_file = feature_folder / f'{feature_base}_component_influence_boxplots.png'
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
    print(f"✅ Component-split boxplot visualizations saved to: {base_folder}")
    print(f"✅ Ready for academic thesis presentation!")
    print(f"{'='*80}\n")

def viz_feature_groups_by_component(
    features_df,
    base_output_folder=Path('./'),
    importance_col='importance',
    component_col='component_name'
):
    setup_professional_style()

    df = features_df.copy()

    if importance_col not in df.columns:
        raise ValueError(f"Column '{importance_col}' not found")

    df['influence_sign'] = np.where(df[importance_col] >= 0, 'positive', 'negative')

    components = ['vocals', 'bass', 'drums', 'other']
    print(f"Processing {len(components)} components...")

    for comp_base in components:
        df['component_base'] = (
            df[component_col].astype(str).str.extract(r'([A-Za-z]+)', expand=False)
        )
        comp_df = df[df['component_base'] == comp_base].copy()

        if comp_df.empty:
            print(f"⚠️ Skipping empty component: {comp_base}")
            continue

        comp_base_folder = Path(base_output_folder) / 'by_component_feature_groups' / comp_base
        comp_base_folder.mkdir(parents=True, exist_ok=True)

        models = sorted(comp_df['model'].dropna().unique())
        signs = ['negative', 'positive']

        exclude_cols = {
            'model', 'track', 'component_key', 'data_type',
            importance_col, 'influence_sign', 'component_base'
        }
        band_meta_cols = {
            'component', 'importance', 'abs_importance',
            'low_freq', 'high_freq', 'band_type', 'model', 'track_stem'
        }

        all_cols = [
            col for col in comp_df.columns
            if (
                col not in exclude_cols and
                col not in band_meta_cols and
                comp_df[col].dtype in ['float64', 'float32', 'int64', 'int32'] and
                comp_df[col].notna().sum() > 0
            )
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

        print(f"  {comp_base}: {len(feature_groups)} feature groups")

        for feature_base, columns_list in sorted(feature_groups.items()):
            feature_folder = comp_base_folder / feature_base
            feature_folder.mkdir(parents=True, exist_ok=True)

            if len(columns_list) == 1 and columns_list[0][1] == 'single':
                col = columns_list[0][0]
                _viz_single_feature_in_component(comp_df, col, feature_folder, models, signs, feature_base)

            else:
                stat_order = ['min', 'mean', 'std', 'max']
                columns_sorted = sorted(
                    columns_list,
                    key=lambda x: next((i for i, stat in enumerate(stat_order) if stat == x[1]), 999)
                )

                for col, stat in columns_sorted:
                    stat_label = stat.upper()
                    _viz_single_feature_in_component(comp_df, col, feature_folder, models, signs, f"{feature_base}_{stat_label}")

        print(f"  ✅ {comp_base_folder} done")

def _viz_single_feature_in_component(comp_df, col, feature_folder, models, signs, feature_base):
    """
    Prywatna: jeden wykres dla jednej cechy w jednym komponencie
    """
    feat_df = comp_df[comp_df[col].notna()].copy()
    if feat_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax_models, ax_global = axes

    # ===== Left panel: per model =====
    plot_data = []
    x_labels = []

    for model in models:
        for s in signs:
            mask = (feat_df['model'] == model) & (feat_df['influence_sign'] == s)
            data = feat_df.loc[mask, col].dropna()
            if len(data) > 0:
                plot_data.append(data.values)
                x_labels.append(f'{model}\n{s}')

    non_empty_indices = [i for i, d in enumerate(plot_data) if len(d) > 0]
    if not non_empty_indices:
        plt.close(fig)
        return

    plot_data = [plot_data[i] for i in non_empty_indices]
    x_labels = [x_labels[i] for i in non_empty_indices]

    bp = ax_models.boxplot(
        plot_data, tick_labels=x_labels, patch_artist=True,
        widths=0.6, showmeans=True, vert=True, whis=1.5,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=7),
        medianprops=dict(color='darkblue', linewidth=2),
        boxprops=dict(linewidth=1.5, color='black')
    )

    for i, patch in enumerate(bp['boxes']):
        s = 'positive' if 'positive' in x_labels[i] else 'negative'
        color = SIGN_COLORS.get(s, '#cccccc')
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)

    for i, data in enumerate(plot_data):
        x = np.random.normal(i + 1, 0.05, size=len(data))
        ax_models.scatter(x, data, alpha=0.35, s=25, c='black', ec='gray')

    ax_models.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax_models.set_title(
        f'{feature_base} per model\n(positive vs negative)',
        fontsize=13, fontweight='bold', pad=15
    )
    ax_models.grid(axis='y', alpha=0.3, ls='--')
    ax_models.spines['top'].set_visible(False)
    ax_models.spines['right'].set_visible(False)
    for tick in ax_models.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')

    # ===== Right panel: global =====
    global_plot_data = []
    global_labels = []

    for s in ['negative', 'positive']:
        data = feat_df[feat_df['influence_sign'] == s][col].dropna()
        if len(data) > 0:
            global_plot_data.append(data.values)
            global_labels.append(s)

    if global_plot_data:
        bp2 = ax_global.boxplot(
            global_plot_data, tick_labels=global_labels, patch_artist=True,
            widths=0.6, showmeans=True, vert=True, whis=1.5,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=7),
            medianprops=dict(color='darkblue', linewidth=2),
            boxprops=dict(linewidth=1.5, color='black')
        )

        for i, patch in enumerate(bp2['boxes']):
            color = SIGN_COLORS.get(global_labels[i], '#cccccc')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

        for i, data in enumerate(global_plot_data):
            x = np.random.normal(i + 1, 0.05, size=len(data))
            ax_global.scatter(x, data, alpha=0.35, s=25, c='black', ec='gray')

        ax_global.set_ylabel('Value', fontsize=13, fontweight='bold')
        ax_global.set_title(
            f'{feature_base}\n(all models, positive vs negative)',
            fontsize=13, fontweight='bold', pad=15
        )
        ax_global.grid(axis='y', alpha=0.3, ls='--')
        ax_global.spines['top'].set_visible(False)
        ax_global.spines['right'].set_visible(False)
    else:
        ax_global.text(
            0.5, 0.5,
            'No data for positive / negative influence',
            transform=ax_global.transAxes,
            ha='center', va='center', fontsize=12, color='red'
        )
        ax_global.axis('off')

    fig.suptitle(f'Component analysis – {feature_base}', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

    stats_text = format_influence_statistics_box(x_labels, plot_data)
    add_bottom_stats_panel(fig, ax_models, stats_text, width_frac=0.45)

    global_stats_text = format_influence_statistics_box(global_labels, global_plot_data)
    add_bottom_stats_panel(fig, ax_global, global_stats_text, width_frac=0.30, y_margin=0.04)

    out_file = feature_folder / f'{feature_base}_most_influential_pos_neg.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f" ✓ {out_file}")

def main():
    args = parse_args()
    config = load_yaml(Path(args.config))

    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    data_root = Path(data_cfg.get("features_path"))
    result_root = Path(output_cfg.get("result_path"))

    features_path = data_root / "audiolime_component_features.json"
    output_root = result_root / "features_visualization"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Output root: {output_root}")
    print("Visualizing AudioLIME component feature importance results")
    print("=" * 70)

    features_df, features_to_analyze = load_and_prepare_data_full(features_path)

    viz_component_pos_neg_boxplots(
        features_df,
        base_output_folder=output_root,
    )

    viz_feature_groups_by_component(
        features_df,
        base_output_folder=output_root
    )

if __name__ == "__main__":
    main()