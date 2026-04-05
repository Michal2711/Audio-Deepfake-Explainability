import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import yaml
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

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

def try_num(s):
    if isinstance(s, bytes):
        s = s.decode('utf-8', errors='ignore')
    match = re.match(r'^(\d+)', s)
    return int(match.group(1)) if match else 999999

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

def safe_reindex_fill(df, index_col, target_idx):
    df_sorted = df.sort_values(index_col)
    df_reidx = df_sorted.set_index(index_col).reindex(target_idx, method='ffill')
    df_final = df_reidx.ffill().reset_index()
    df_final.rename(columns={'index': index_col}, inplace=True)
    return df_final

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

def load_audiolime_explanations(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    components = ['vocals0', 'drums0', 'bass0', 'other0']
    
    for model_name, tracks_dict in data.items():
        for track_key, track_data in tracks_dict.items():
            if not isinstance(track_data, dict):
                continue
                
            expl = track_data.get("explanations", {})
            comp_inf = expl.get("component_influences", {})
            track_id = track_data.get("track_id")
            
            pred_score = float(expl.get("model_prediction", float("nan")))
            
            for comp_name in components:
                rows.append({
                    "model": model_name,
                    "track": track_key,
                    "track_id": track_id,
                    "component_name": comp_name,
                    "prediction_score": pred_score,
                    "predicted_class": expl.get("predicted_class"),
                    f"{comp_name}_influence": float(comp_inf.get(comp_name, float("nan"))),
                })
    
    lime_df = pd.DataFrame(rows)
    print(f"AudioLIME explanations: {len(lime_df)} rows")
    print("Lime columns:", lime_df.columns.tolist())
    return lime_df

def format_influence_statistics_box(labels, plot_data):
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

def plot_audiolime_predictions_influence_features(features_df: pd.DataFrame, lime_json_path: Path, outputdir: Path):
    setup_professional_style()
    sns.set_theme(style='whitegrid')
    
    lime_df = load_audiolime_explanations(lime_json_path)
    
    full_df = pd.merge(
        features_df, 
        lime_df, 
        on=["model", "track", "component_name"],
        how="inner"
    )
    
    print(f"✅ Merged AudioLIME full_df: {len(full_df)} rows")
    print("Sample columns:", full_df[['model', 'track', 'component_name', 'prediction_score', 'vocals0_influence', 'rms_wave_mean']].head(2).to_string())
    
    outputdir = outputdir / 'audiolime_3rows_per_component'
    outputdir.mkdir(parents=True, exist_ok=True)
    
    exclude_cols = ['model', 'track', 'track_id', 'datatype', 'component_name', 'component_type', 
                    'prediction_score', 'predicted_class', 'vocals0_influence', 'drums0_influence', 
                    'bass0_influence', 'other0_influence', 
                    'abs_importance', 'importance'
        ]
    feature_cols = [c for c in full_df.columns if pd.api.types.is_numeric_dtype(full_df[c]) and c not in exclude_cols]
    
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
    
    components = ['vocals0', 'drums0', 'bass0', 'other0']
    models = full_df['model'].unique()
    
    for model in sorted(models):
        model_dir = outputdir / model.replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for comp in components:
            comp_dir = model_dir / comp
            comp_dir.mkdir(parents=True, exist_ok=True)
            comp_df = full_df[(full_df['model'] == model) & (full_df['component_name'] == comp)]
            if len(comp_df) == 0:
                print(f"⚠️ Skip {model}/{comp}: brak danych")
                continue
            
            comp_df = comp_df.reset_index(drop=True)
            comp_df['file_index'] = range(len(comp_df))
            n_files = len(comp_df)
            
            track_stems = comp_df['track_stem'].tolist()
            
            comp_cols = {
                'vocals0': 'vocals0_influence',
                'drums0': 'drums0_influence', 
                'bass0': 'bass0_influence',
                'other0': 'other0_influence'
            }
            comp_influence_col = comp_cols[comp]
            
            comp_features = [c for c in feature_cols if comp_df[c].notna().sum() > 0]
            
            for feat_col in comp_features:
                if feat_col not in comp_df.columns or comp_df[feat_col].isna().all():
                    continue
                
                base_feat_name = re.sub(r'_(mean|std|min|max)$', '', feat_col)
                base_feat_dir = comp_dir / base_feat_name
                base_feat_dir.mkdir(exist_ok=True, parents=True)

                d_pred = comp_df[['file_index', 'prediction_score']].dropna()
                d_lime = comp_df[['file_index', comp_influence_col]].dropna()
                d_feat = comp_df[['file_index', feat_col]].dropna()
                
                all_idx = range(n_files)
                d_pred = safe_reindex_fill(d_pred, 'file_index', all_idx)
                d_lime = safe_reindex_fill(d_lime, 'file_index', all_idx)
                d_feat = safe_reindex_fill(d_feat, 'file_index', all_idx)

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, height_ratios=[1, 1, 1])
                
                ax1.plot(d_pred['file_index'], d_pred['prediction_score'], 'o-', linewidth=3, markersize=8,
                        color='darkred', alpha=0.9, label='Predictions (P>0.5=Fake)')
                ax1.axhline(y=0.5, color='black', ls='-', lw=2.5, alpha=0.8)
                ax1.set_ylabel('Predictions', fontweight='bold', fontsize=12)
                ax1.grid(alpha=0.3, ls='--')
                ax1.legend(loc='upper right')
                ax1.set_title(f'{model} | {comp}', fontsize=14, fontweight='bold', pad=20)
                
                ax2.plot(d_lime['file_index'], d_lime[comp_influence_col], 'o-', linewidth=3, markersize=8,
                        color='purple', alpha=0.85, label=f'LIME: {comp}')
                ax2.axhline(y=0, color='gray', ls=':', lw=2)
                ax2.set_ylabel('LIME Influence', fontweight='bold', fontsize=12)
                ax2.grid(alpha=0.3, ls='--')
                ax2.legend(loc='upper right')
                
                ax3.plot(d_feat['file_index'], d_feat[feat_col], 'o-', linewidth=3, markersize=8,
                        color='steelblue', alpha=0.85, label=feat_col.replace('_', ' ').title())
                ax3.set_xlabel('File Index in Component')
                ax3.set_ylabel('Physical Feature', fontweight='bold', fontsize=12)
                ax3.grid(alpha=0.3, ls='--')
                ax3.legend(loc='upper right')
                
                plt.tight_layout()
                
                short_labels = [f"{i}: {stem[:25]}..." for i, stem in enumerate(track_stems)]
                fig.text(1.015, 0.5, 'File Mapping:\n' + '\n'.join(short_labels[:10]), 
                        fontsize=17, va='center', ha='left',
                        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.95))
                
                safe_feat = re.sub(r'[^a-zA-Z0-9_]', '_', feat_col)
                safe_comp = comp.replace('0', '')
                outfile = base_feat_dir / f"{safe_comp}_{safe_feat}_3rows.png"
                plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"✓ {model}/{safe_comp}_{safe_feat}_3rows.png")

def plot_audiolime_3rows_multicolumn(features_df: pd.DataFrame, lime_json_path: Path, outputdir: Path):
    setup_professional_style()
    sns.set_theme(style='whitegrid')
    
    lime_df = load_audiolime_explanations(lime_json_path)
    full_df = pd.merge(features_df, lime_df, on=["model", "track", "component_name"], how="inner")
    
    outputdir = outputdir / 'audiolime_3rows_multicolumn'
    outputdir.mkdir(parents=True, exist_ok=True)
    
    components = ['vocals0', 'drums0', 'bass0', 'other0']
    comp_cols = {'vocals0': 'vocals0_influence', 'drums0': 'drums0_influence', 
                 'bass0': 'bass0_influence', 'other0': 'other0_influence'}
    comp_names = ['Vocals', 'Drums', 'Bass', 'Other']
    n_comp = len(components)
    
    exclude_cols = ['model', 'track', 'track_id', 'datatype', 'component_name', 'component_type', 
                    'prediction_score', 'predicted_class', 'vocals0_influence', 'drums0_influence', 
                    'bass0_influence', 'other0_influence', 
                    'abs_importance', 'importance'] + list(comp_cols.values())
    feature_cols = [c for c in full_df.columns if pd.api.types.is_numeric_dtype(full_df[c]) and c not in exclude_cols]
    
    print(f"All feature cols: {len(feature_cols)}")
    
    models = sorted(full_df['model'].unique())
    
    for model in models:
        model_df = full_df[full_df['model'] == model]
        model_dir = outputdir / model.replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        track_stems = sorted(model_df['track_stem'].dropna().unique(), key=try_num)
        file_idx_map = {stem: i for i, stem in enumerate(track_stems)}
        n_files = len(track_stems)
        
        model_pred = model_df.groupby('track_stem')['prediction_score'].mean()
        pred_values = [model_pred.get(stem, np.nan) for stem in track_stems]
        
        for feat_col in feature_cols:
            if model_df[feat_col].isna().all():
                continue
        
            base_feat_name = re.sub(r'_(mean|std|min|max)$', '', feat_col)
            
            fig, axes = plt.subplots(3, n_comp, figsize=(5*n_comp, 12), 
                                   sharex='col', sharey='row')
            if n_comp == 1:
                axes = axes.reshape(3, 1)
            
            for j in range(n_comp):
                axes[0, j].plot(range(n_files), pred_values, 'o-', lw=2.5, ms=6, 
                               color='darkred', alpha=0.9)
                axes[0, j].axhline(0.5, color='black', ls='-', lw=2, alpha=0.8)
                axes[0, j].set_title('Predictions', fontweight='bold')
                axes[0, j].grid(alpha=0.3, ls='--')
            
            for j, comp in enumerate(components):
                comp_data = model_df[model_df['component_name'] == comp]
                
                lime_vals = comp_data[comp_cols[comp]].tolist()
                if len(lime_vals) > 0:
                    axes[1, j].plot(range(min(n_files, len(lime_vals))), lime_vals[:n_files], 
                                   'o-', lw=2.5, ms=6, color='purple', alpha=0.85)
                    axes[1, j].axhline(0, color='gray', ls=':', lw=2)
                axes[1, j].set_title(f'{comp_names[j]} LIME', fontweight='bold')
                axes[1, j].grid(alpha=0.3, ls='--')
                
                feat_vals = comp_data[feat_col].tolist()
                if len(feat_vals) > 0:
                    axes[2, j].plot(range(min(n_files, len(feat_vals))), feat_vals[:n_files], 
                                   'o-', lw=2.5, ms=6, color='steelblue', alpha=0.85)
                axes[2, j].set_title(f'{comp_names[j]} {feat_col.replace("_", " ").title()}', fontweight='bold')
                axes[2, j].grid(alpha=0.3, ls='--')
                axes[2, j].set_xlabel('File Index')
            
            plt.suptitle(f'{model}: {feat_col.replace("_", " ").title()} | All components', 
                        fontsize=16, y=0.98)
            plt.tight_layout()
            
            short_labels = [f"{i}: {s[:25]}..." for i, s in enumerate(track_stems)]
            fig.text(1.02, 0.48, 'File Mapping:\n' + '\n'.join(short_labels[:10]), 
                    fontsize=17, va='center', ha='left',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.95))
            
            safe_feat = re.sub(r'[^a-zA-Z0-9_]', '_', feat_col)
            base_feat_dir = model_dir / base_feat_name
            base_feat_dir.mkdir(exist_ok=True, parents=True)
            outfile = base_feat_dir / f"{safe_feat}_all_components.png"

            plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"✓ {model}/{safe_feat}_all_components.png")

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

def viz_single_feature_vs_importance_in_component(
    component_df,
    feature_col,
    feature_folder,
    feature_label,
    importance_col="importance",
    sign_col="influence_sign",
):
    required_cols = [feature_col, importance_col, "model", sign_col]
    for col in required_cols:
        if col not in component_df.columns:
            return

    base_df = component_df[required_cols].dropna(
        subset=[feature_col, importance_col]
    ).copy()
    if base_df.empty:
        return

    models = sorted(base_df["model"].dropna().unique().tolist())
    if not models:
        return

    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(4 * n_models, 6), sharey=True
    )
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_df = base_df[base_df["model"] == model_name]
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
    fig.supylabel("Component importance", fontsize=13, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis='x', labelrotation=45, labelsize=10)
        ax.xaxis.set_tick_params(labelbottom=True)

    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D(
            [0], 
            [0], 
            marker='o', 
            color='w', 
            markerfacecolor='blue', 
            markeredgecolor='black', 
            label='positive', 
            markersize=8
        ),
        Line2D(
            [0], 
            [0], 
            marker='X', 
            color='w',
            markerfacecolor='blue', 
            markeredgecolor='black', 
            label='negative', 
            markersize=8, 
            linewidth=1.8
        ),
    ]
    fig.legend(handles=legend_elements, 
               title="Signs of influence", 
               loc='upper right', 
               bbox_to_anchor=(0.98, 0.98), 
               fontsize=9, 
               title_fontsize=10)


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

def viz_feature_values_vs_importance_by_component(
    features_df,
    base_output_folder=Path("."),
    importance_col="importance",
    component_col="component_name",
):
    setup_professional_style()
    df = features_df.copy()

    if importance_col not in df.columns:
        raise ValueError(f"Column '{importance_col}' not found")

    df["influence_sign"] = np.where(df[importance_col] >= 0, "positive", "negative")

    df["component_base"] = (
        df[component_col].astype(str).str.extract(r"([A-Za-z]+)", expand=False)
    )

    components = ["vocals", "bass", "drums", "other"]
    print(f"Processing {len(components)} components for feature vs importance...")

    for comp_base in components:
        comp_df = df[df["component_base"] == comp_base].copy()
        if comp_df.empty:
            print(f"Skipping empty component {comp_base}")
            continue

        comp_base_folder = (
            Path(base_output_folder)
            / "by_component_feature_vs_importance"
            / comp_base
        )
        comp_base_folder.mkdir(parents=True, exist_ok=True)

        models = sorted(comp_df["model"].dropna().unique().tolist())

        exclude_cols = [
            "model",
            "track",
            "component_key",
            "data_type",
            importance_col,
            "influence_sign",
            "component_base",
        ]
        band_meta_cols = [
            "component",
            "importance",
            "abs_importance",
            "low_freq",
            "high_freq",
            "band_type",
            "model",
            "track_stem",
        ]

        allcols = [
            col
            for col in comp_df.columns
            if col not in exclude_cols
            and col not in band_meta_cols
            and comp_df[col].dtype in (np.float64, np.float32, np.int64, np.int32)
            and comp_df[col].notna().sum() > 0
        ]

        feature_groups = defaultdict(list)
        for col in allcols:
            parts = col.split("_")
            if len(parts) > 1 and parts[-1] in ("min", "mean", "std", "max"):
                base_name = "_".join(parts[:-1])
                stat = parts[-1]
            else:
                base_name = col
                stat = "single"
            feature_groups[base_name].append((col, stat))

        print(f"{comp_base}: {len(feature_groups)} feature groups")

        for feature_base, columns_list in sorted(feature_groups.items()):
            feature_folder = comp_base_folder / feature_base
            feature_folder.mkdir(parents=True, exist_ok=True)

            if len(columns_list) == 1 and columns_list[0][1] == "single":
                col, _ = columns_list[0]
                feature_label = feature_base
                viz_single_feature_vs_importance_in_component(
                    comp_df,
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
                    viz_single_feature_vs_importance_in_component(
                        comp_df,
                        col,
                        feature_folder,
                        feature_label,
                        importance_col=importance_col,
                        sign_col="influence_sign",
                    )

        print(f"{comp_base_folder} done")

MODEL_ORDER = ['ElevenLabs', 'REAL', 'SUNO', 'SUNO_PRO', 'UDIO']

TYPE_MAPPING = {
    'ElevenLabs': 'GENERATED',
    'REAL': 'REAL',
    'SUNO': 'GENERATED',
    'SUNO_PRO': 'GENERATED',
    'UDIO': 'GENERATED',
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

def _assign_feature_group(col: str) -> str:
    """Return the semantic group name for a feature column, or 'other' if unmatched."""
    for group, prefixes in FEATURE_GROUPS_DEF.items():
        for prefix in prefixes:
            if col.startswith(prefix):
                return group
    return 'other'


def _build_corr_matrix(comp_df, feature_cols, target_col, groups_bool):
    """Build (features × groups) Pearson r DataFrame."""
    r_dict = {}
    for group_label, mask in groups_bool.items():
        gdf = comp_df[mask]
        r_vals = {}
        for feat in feature_cols:
            sub = gdf[[feat, target_col]].dropna()
            r_vals[feat] = sub[feat].corr(sub[target_col]) if len(sub) >= 3 else np.nan
        r_dict[group_label] = r_vals

    r_df = pd.DataFrame(r_dict)
    r_df = r_df.dropna(how='all')
    if not r_df.empty:
        stat_order = {'mean': 0, 'std': 1, 'min': 2, 'max': 3}
        stat_suffix = re.compile(r'_(mean|std|min|max)$')

        def base_name(col):
            return stat_suffix.sub('', col)

        def stat_rank(col):
            m = stat_suffix.search(col)
            return stat_order.get(m.group(1), 99) if m else -1

        r_df['_base'] = [base_name(c) for c in r_df.index]
        base_importance = (
            r_df.drop(columns='_base')
            .abs().max(axis=1)
            .groupby(r_df['_base'])
            .transform('max')
        )
        r_df['_base_importance'] = base_importance
        r_df['_stat_rank']       = [stat_rank(c) for c in r_df.index]

        r_df = (
            r_df.sort_values(['_base_importance', '_base', '_stat_rank'],
                             ascending=[False, True, True])
            .drop(columns=['_base', '_base_importance', '_stat_rank'])
        )
    return r_df


def _save_corr_heatmap(r_df, title, outfile):
    """Render and save a seaborn coolwarm heatmap."""
    if r_df.empty:
        print(f'  ⚠️  Empty r_df – skipping {outfile.name}')
        return

    n_feats = len(r_df)
    n_cols  = len(r_df.columns)
    fig_h   = max(4, n_feats * 0.42 + 2.5)
    fig_w   = max(10, n_cols * 1.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    mask_nan = r_df.isnull()

    sns.heatmap(
        r_df, ax=ax,
        cmap='coolwarm', vmin=-1, vmax=1,
        annot=True, fmt='.2f',
        linewidths=0.4, linecolor='#dddddd',
        mask=mask_nan,
        cbar_kws={'label': 'Pearson r', 'shrink': 0.6},
        annot_kws={'size': 8, 'weight': 'bold'},
    )
    ax.patch.set_facecolor('#f0f0f0')   # grey for NaN cells
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Group', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=10)
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ {outfile.relative_to(outfile.parents[4]) if outfile.parents[4].exists() else outfile.name}')

def plot_feature_correlation_r_heatmaps(
    features_df: pd.DataFrame,
    lime_json_path: Path,
    comp_version: str,
    outputdir: Path,
    model_order: list = None,
) -> None:
    """
    Per component (vocals, drums, bass, other) × semantic feature group
    create two heatmap PNGs:
      1. Physical features  -  prediction score P(fake)
      2. Physical features  -  LIME component influence

    Heatmap layout
    ──────────────
    Rows    : feature names within the group (sorted by max |r|)
    Columns : generated | real | ElevenLabs | REAL | SUNO | SUNO_PRO | UDIO
    Colour  : Pearson r  (coolwarm, −1 … +1)

    Output structure
    ────────────────
    <outputdir>/correlation_r_heatmaps/
      vocals/
        [Group_name]/
          vocals_[Group_name]_r_vs_prediction.png
          vocals_[Group_name]_r_vs_lime_influence.png
        ...
      drums/  ...
      bass/   ...
      other/  ...
    """
    setup_professional_style()
    sns.set_theme(style='whitegrid')

    if model_order is None:
        model_order = MODEL_ORDER

    lime_df = load_audiolime_explanations(lime_json_path)
    full_df = pd.merge(features_df, lime_df, on=['model', 'track', 'component_name'], how='inner')

    if full_df.empty:
        print('⚠️  Merged DataFrame is empty.')
        return

    print(f'\n{"="*70}')
    print('Generating feature correlation r heatmaps (split by feature group)…')
    print(f'  Merged rows : {len(full_df)}')
    print(f'{"="*70}\n')

    full_df['data_type_str'] = full_df['model'].map(TYPE_MAPPING).fillna('GENERATED')

    feature_cols = [
        c for c in full_df.columns
        if c not in _CORR_EXCLUDE
        and pd.api.types.is_numeric_dtype(full_df[c])
        and full_df[c].notna().sum() > 0
    ]

    feat_to_group = {c: _assign_feature_group(c) for c in feature_cols}
    groups_present = sorted(set(feat_to_group.values()))
    print(f'  Feature groups found: {groups_present}')
    for g in groups_present:
        n = sum(1 for v in feat_to_group.values() if v == g)
        print(f'    {g}: {n} features')

    model_cols = [(m, full_df['model'] == m) for m in model_order if m in full_df['model'].unique()]
    group_defs_ordered = (
        [('all',       pd.Series(True, index=full_df.index)),
         ('generated', full_df['data_type_str'] == 'GENERATED'),
         ('real',      full_df['data_type_str'] == 'REAL')]
        + model_cols
    )

    components = ['vocals0', 'drums0', 'bass0', 'other0']
    comp_influence_cols = {
        'vocals0': 'vocals0_influence',
        'drums0':  'drums0_influence',
        'bass0':   'bass0_influence',
        'other0':  'other0_influence',
    }

    root_out = outputdir / 'correlation_r_heatmaps'
    root_out.mkdir(parents=True, exist_ok=True)

    for comp in components:
        comp_name = comp.replace('0', '')
        comp_dir  = root_out / comp_name
        comp_dir.mkdir(parents=True, exist_ok=True)

        comp_df = full_df[full_df['component_name'] == comp].copy()
        if comp_df.empty:
            print(f'  ⏭️  No data for {comp}')
            continue

        print(f'\n  Component: {comp_name.upper()}  ({len(comp_df)} rows)')

        groups_bool = {}
        for label, mask in group_defs_ordered:
            m = mask.reindex(comp_df.index, fill_value=False)
            groups_bool[label] = m

        influence_col = comp_influence_cols[comp]

        targets = [
            ('prediction_score', 'Prediction P(fake)',   'r_vs_prediction'),
            (influence_col,       f'LIME influence ({comp_name})', 'r_vs_lime_influence'),
        ]

        for feat_group in groups_present:
            grp_feats = [c for c, g in feat_to_group.items() if g == feat_group]
            grp_feats = [c for c in grp_feats if comp_df[c].notna().sum() >= 3]
            if not grp_feats:
                continue

            grp_dir = comp_dir / feat_group
            grp_dir.mkdir(parents=True, exist_ok=True)

            for target_col, target_label, suffix in targets:
                if target_col not in comp_df.columns:
                    continue

                r_df = _build_corr_matrix(comp_df, grp_feats, target_col, groups_bool)
                if r_df.empty:
                    continue

                if comp_version == 'separated':
                    title = (
                        f'{comp_name.capitalize()} | {feat_group.replace("_", " ").title()}\n' 
                        f'Pearson r  ←→  {target_label}'
                    )
                elif comp_version == 'reversed':
                    title = (
                        f'Mix - {comp_name.capitalize()} | {feat_group.replace("_", " ").title()}\n' 
                        f'Pearson r  ←→  {target_label}'
                    )
                fname = grp_dir / f'{comp_name}_{feat_group}_{suffix}.png'
                _save_corr_heatmap(r_df, title, fname)

        all_feats = [c for c in feature_cols if comp_df[c].notna().sum() >= 3]
        for target_col, target_label, suffix in targets:
            if target_col not in comp_df.columns:
                continue

            r_df_all = _build_corr_matrix(comp_df, all_feats, target_col, groups_bool)
            if r_df_all.empty:
                continue

            if 'all' in r_df_all.columns:
                r_df_all = r_df_all.reindex(
                    r_df_all['all'].abs().sort_values(ascending=False).index
                )

            if comp_version == 'separated':
                title = (
                    f'{comp_name.capitalize()} | All features\n'
                    f'Pearson r  ←→  {target_label}  (sorted by overall |r|)'
                )
            elif comp_version == 'reversed':
                title = (
                    f'Mix - {comp_name.capitalize()} | All features\n'
                    f'Pearson r  ←→  {target_label}  (sorted by overall |r|)'
                )
            fname = comp_dir / f'{comp_name}_all_features_{suffix}.png'
            _save_corr_heatmap(r_df_all, title, fname)

_TBL_BG         = '#0e1117'
_TBL_HEADER_BG  = '#1a1d27'
_TBL_ROW_ALT_BG = '#13161f'
_TBL_TEXT       = '#d0d0d0'
_TBL_HEADER_TXT = '#7a8099'
_POS_STRONG     = '#ff6b35'
_POS_MEDIUM     = '#e8943a'
_NEG_STRONG     = '#2ecc71'
_NEG_MEDIUM     = '#27ae60'
_NEAR_ZERO      = '#8899aa'


def _tbl_fmt_value(v):
    if pd.isna(v):     return '—'
    av = abs(v)
    if av == 0:        return '0'
    if av >= 1000:     return f'{v:,.0f}'
    if av >= 10:       return f'{v:.2f}'
    if av >= 1:        return f'{v:.3f}'
    if av >= 0.001:    return f'{v:.4f}'
    return f'{v:.2e}'


def _tbl_fmt_pct(pct):
    if pd.isna(pct) or abs(pct) <= 5:
        return '(≈0%)'
    sign = '+' if pct > 0 else ''
    return f'({sign}{pct:.0f}%)'


def _tbl_pct_color(pct):
    if pd.isna(pct) or abs(pct) <= 5:
        return _NEAR_ZERO
    if pct > 0:
        return _POS_STRONG if abs(pct) > 30 else _POS_MEDIUM
    return _NEG_STRONG if abs(pct) > 30 else _NEG_MEDIUM


def _draw_comparison_table(
    feat_list, real_vals, means_v, pct_df, sources,
    title_str, outfile,
    figsize_w=14.0, row_height=0.40, dpi=180,
    col_header_colors=None,
    strip_stat_suffix=True,
):
    """
    Render and save one dark-themed comparison table.
    """
    import matplotlib.patches as mpatches

    n_rows  = len(feat_list)
    n_cols  = 2 + len(sources)
    fig_h   = max(4.0, n_rows * row_height + 1.8)

    fig = plt.figure(figsize=(figsize_w, fig_h), facecolor=_TBL_BG)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(_TBL_BG)
    ax.axis('off')

    col_labels = ['FEATURE', 'REAL'] + sources
    raw_widths = [0.30] + [0.12] * (n_cols - 1)
    tot_w      = sum(raw_widths)
    col_widths = [w / tot_w for w in raw_widths]
    col_lefts  = []
    x = 0.01
    for w in col_widths:
        col_lefts.append(x)
        x += w * (0.99 / tot_w * tot_w)

    def _cell(ridx, cidx, text, color=_TBL_TEXT,
              bg=_TBL_BG, fs=8.5, bold=False, align='right'):
        x0 = col_lefts[cidx]
        cw = col_widths[cidx]
        y0 = 1.0 - (ridx + 1) * (1.0 / (n_rows + 2))
        ch = 1.0 / (n_rows + 2)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), cw, ch,
            boxstyle='square,pad=0', linewidth=0,
            facecolor=bg, transform=ax.transAxes, clip_on=False,
        ))
        tx = x0 + cw * (0.95 if align == 'right' else 0.05)
        ax.text(
            tx, y0 + ch * 0.5, text,
            color=color, fontsize=fs,
            ha=align, va='center',
            fontweight='bold' if bold else 'normal',
            transform=ax.transAxes, clip_on=False,
            fontfamily='monospace',
        )

    for ci, lbl in enumerate(col_labels):
        hdr_color = (
            (col_header_colors or {}).get(lbl, _TBL_HEADER_TXT)
        )
        _cell(0, ci, lbl.upper().replace('_', ' '),
              color=hdr_color, bg=_TBL_HEADER_BG,
              fs=8, bold=True, align='left' if ci == 0 else 'right')

    for ri, feat in enumerate(feat_list, start=1):
        row_bg = _TBL_ROW_ALT_BG if ri % 2 == 0 else _TBL_BG
        disp = (
            re.sub(r'_(mean|std|min|max)$', '', feat)
            if strip_stat_suffix
            else feat
        ).replace('_', ' ').title()
        _cell(ri, 0, disp, color=_TBL_TEXT, bg=row_bg, align='left')
        _cell(ri, 1, _tbl_fmt_value(real_vals[feat]),
              color=_TBL_TEXT, bg=row_bg)
        for si, src in enumerate(sources):
            sv  = means_v.loc[src, feat] if src in means_v.index else np.nan
            pct = pct_df.loc[feat, src]  if src in pct_df.columns else np.nan
            txt = f'{_tbl_fmt_value(sv)}  {_tbl_fmt_pct(pct)}'
            _cell(ri, 2 + si, txt, color=_tbl_pct_color(pct), bg=row_bg)

    ax.text(0.01, 0.995, title_str,
            color='#aabbcc', fontsize=9.5, fontweight='bold',
            ha='left', va='top', transform=ax.transAxes, fontfamily='monospace')

    legend = [
        (_POS_STRONG, '> +30%'), (_POS_MEDIUM, '+15–30%'),
        (_NEAR_ZERO,  '≈ 0%'),
        (_NEG_MEDIUM, '−15–30%'), (_NEG_STRONG, '< −30%'),
    ]
    lx = 0.01
    ax.text(lx, 0.008, 'Deviation from REAL:', color=_TBL_HEADER_TXT,
            fontsize=7, ha='left', va='bottom', transform=ax.transAxes)
    lx += 0.16
    for col, lbl in legend:
        ax.text(lx, 0.008, f'■ {lbl}', color=col, fontsize=7,
                ha='left', va='bottom', transform=ax.transAxes,
                fontfamily='monospace')
        lx += 0.10

    plt.savefig(outfile, dpi=dpi, bbox_inches='tight',
                facecolor=_TBL_BG, edgecolor='none')
    plt.close()
    print(f'  ✅ {outfile}')



def _build_pred_split(comp_df, feat_cols, sources, model_col,
                      pred_col='predicted_class',
                      real_label='Real', fake_label='Fake'):
    compound_sources = []
    rows = {}
    col_header_colors = {}
    for src in sources:
        src_df = comp_df[comp_df[model_col] == src]
        for pred_label, color in [(real_label, _NEG_MEDIUM), (fake_label, _POS_STRONG)]:
            key = f'{src} [{pred_label}]'
            compound_sources.append(key)
            col_header_colors[key] = color
            subset = src_df[src_df[pred_col] == pred_label]
            rows[key] = (
                subset[feat_cols].mean()
                if not subset.empty
                else pd.Series(np.nan, index=feat_cols)
            )
    means_split = pd.DataFrame(rows).T
    return means_split, compound_sources, col_header_colors


def plot_feature_comparison_table(
    features_df: pd.DataFrame,
    lime_json_path: Path,
    comp_version: str,
    outputdir: Path,
    component_col: str = 'component_name',
    model_col: str = 'model',
    model_order: list = None,
    feature_groups: dict = None,
    multi_stat_groups: list = None,
    sort_by_deviation: bool = True,
    figsize_w: float = 14.0,
    row_height: float = 0.40,
    dpi: int = 180,
) -> None:
    """
    Per component × feature group, create dark-themed comparison table PNGs.

    Structure
    ---------
    <outputdir>/feature_comparison_tables/
      <component>/
        <GroupName>/
          <comp>_<GroupName>.png
        Frequency_spectrum/
          <comp>_Frequency_spectrum_mean.png   <- 4 separate files
          <comp>_Frequency_spectrum_std.png
          <comp>_Frequency_spectrum_min.png
          <comp>_Frequency_spectrum_max.png
    """
    if model_order is None:
        model_order = MODEL_ORDER
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS_DEF
    if multi_stat_groups is None:
        multi_stat_groups = ['Frequency_spectrum']

    root_out = outputdir / 'feature_comparison_tables'
    root_out.mkdir(parents=True, exist_ok=True)

    setup_professional_style()
    lime_df = load_audiolime_explanations(lime_json_path)
    full_df = pd.merge(features_df, lime_df, on=["model", "track", "component_name"], how="inner")

    _meta = {
        model_col, component_col, 'track', 'track_id', 'component_key',
        'data_type', 'data_type_str', 'component_type', 'importance',
        'abs_importance', 'track_stem', 'low_freq', 'high_freq', 'band_type',
        'vocals0_influence', 'drums0_influence', 'bass0_influence', 'other0_influence',
        'prediction_score', 'predicted_class',
    }
    all_feat_cols = [
        c for c in full_df.columns
        if c not in _meta and pd.api.types.is_numeric_dtype(full_df[c])
    ]

    def _feat_group(col):
        for g, prefixes in feature_groups.items():
            for p in prefixes:
                if col.startswith(p):
                    return g
        return 'inne'

    _stat_re = re.compile(r'_(mean|std|min|max)$')
    stat_order_list = ['mean', 'std', 'min', 'max']

    components = ['vocals0', 'drums0', 'bass0', 'other0']

    for comp in components:
        comp_name = comp.replace('0', '')
        comp_df   = full_df[full_df[component_col] == comp]
        if comp_df.empty:
            print(f'  ⏭️  No data for {comp}')
            continue

        means = comp_df.groupby(model_col)[all_feat_cols].mean()
        if 'REAL' not in means.index:
            print(f'  ⚠️  No REAL rows for {comp}')
            continue

        real_vals = means.loc['REAL']
        sources   = [s for s in model_order if s in means.index and s != 'REAL']

        pct_diffs = {}
        for src in sources:
            sv = means.loc[src]
            with np.errstate(divide='ignore', invalid='ignore'):
                pct = np.where(
                    real_vals != 0,
                    (sv - real_vals) / real_vals.abs() * 100,
                    np.nan,
                )
            pct_diffs[src] = pd.Series(pct, index=all_feat_cols)
        pct_df_full = pd.DataFrame(pct_diffs)

        valid     = real_vals.dropna().index
        real_vals = real_vals.loc[valid]
        means_v   = means[valid]
        pct_df_full = pct_df_full.loc[valid]

        comp_root = root_out / comp_name
        comp_root.mkdir(parents=True, exist_ok=True)

        comp_df_full = comp_df.copy()
        all_groups = list(feature_groups.keys()) + ['inne']
        for grp in all_groups:
            grp_feats = [c for c in valid if _feat_group(c) == grp]
            if not grp_feats:
                continue

            grp_dir = comp_root / grp
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
                    
                    if comp_version == 'separated':
                        title = (
                            f'{comp_name.upper()} | {grp.replace("_", " ")} '
                            f'[{stat.upper()}] – mean values vs REAL baseline'
                        )
                    elif comp_version == 'reversed':
                        title = (
                            f'Mix - {comp_name.upper()} | {grp.replace("_", " ")} '
                            f'[{stat.upper()}] – mean values vs REAL baseline'
                        )
                    fname = grp_dir / f'{comp_name}_{grp}_{stat}.png'
                    _draw_comparison_table(
                        stat_feats, real_vals, means_v, pct_df_full, sources,
                        title, fname, figsize_w=figsize_w,
                        row_height=row_height, dpi=dpi,
                    )
                    ms, cs, chc = _build_pred_split(
                        comp_df_full, stat_feats, sources, model_col)
                    if not ms.empty:
                        pct_p = pd.DataFrame({
                            c: np.where(
                                real_vals[stat_feats] != 0,
                                (ms.loc[c] - real_vals[stat_feats])
                                / real_vals[stat_feats].abs() * 100,
                                np.nan,
                            ) if c in ms.index else np.nan
                            for c in cs
                        }, index=stat_feats)
                        title_p = title + '  |  split by prediction'
                        fname_p = grp_dir / f'{comp_name}_{grp}_{stat}_by_pred.png'
                        _draw_comparison_table(
                            stat_feats, real_vals, ms, pct_p, cs,
                            title_p, fname_p,
                            figsize_w=figsize_w * 1.7,
                            row_height=row_height, dpi=dpi,
                            col_header_colors=chc,
                        )
            else:
                if sort_by_deviation:
                    grp_feats = list(
                        pct_df_full.loc[grp_feats].abs()
                        .max(axis=1).sort_values(ascending=False).index
                    )

                if comp_version == 'separated':
                    title = (
                        f'{comp_name.upper()} | {grp.replace("_", " ")} '
                        f'– mean values vs REAL baseline'
                    )
                elif comp_version == 'reversed':
                    title = (
                        f'Mix - {comp_name.upper()} | {grp.replace("_", " ")} '
                        f'– mean values vs REAL baseline'
                    )
                fname = grp_dir / f'{comp_name}_{grp}.png'
                _draw_comparison_table(
                    grp_feats, real_vals, means_v, pct_df_full, sources,
                    title, fname, figsize_w=figsize_w,
                    row_height=row_height, dpi=dpi,
                    strip_stat_suffix=False,
                )
                ms, cs, chc = _build_pred_split(
                    comp_df_full, grp_feats, sources, model_col)
                if not ms.empty:
                    pct_p = pd.DataFrame({
                        c: np.where(
                            real_vals[grp_feats] != 0,
                            (ms.loc[c] - real_vals[grp_feats])
                            / real_vals[grp_feats].abs() * 100,
                            np.nan,
                        ) if c in ms.index else np.nan
                        for c in cs
                    }, index=grp_feats)
                    title_p = title + '  |  split by prediction'
                    fname_p = grp_dir / f'{comp_name}_{grp}_by_pred.png'
                    _draw_comparison_table(
                        grp_feats, real_vals, ms, pct_p, cs,
                        title_p, fname_p,
                        figsize_w=figsize_w * 1.7,
                        row_height=row_height, dpi=dpi,
                        col_header_colors=chc,
                        strip_stat_suffix=False,
                    )

    print(f'\n✅  Comparison tables saved to: {root_out}\n')

def main():
    args = parse_args()
    config = load_yaml(Path(args.config))

    data_cfg = config.get("data", {})
    explanations_cfg = config.get("explanations_data", {})
    output_cfg = config.get("output", {})
    lime_comp_features_cfg = config.get("lime_comp_features", {})
    comp_version = lime_comp_features_cfg.get("version", "separated")

    data_root = Path(data_cfg.get("features_path"))
    explanations_path = explanations_cfg.get("explanations_path")
    result_root = Path(output_cfg.get("result_path"))

    data_root = data_root / "separated_components" if comp_version == "separated" else data_root / "reversed_separated_components"
    features_path = data_root / "lime_components" / "audiolime_component_features.json"

    result_root = result_root / "separated_components" if comp_version == "separated" else result_root / "reversed_separated_components"

    output_root = result_root / "features_visualization"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Output root: {output_root}")
    print("Visualizing AudioLIME component feature importance results")
    print("=" * 70)

    features_df, features_to_analyze = load_and_prepare_data_full(features_path)

    # viz_component_pos_neg_boxplots(
    #     features_df,
    #     base_output_folder=output_root,
    # )

    # viz_feature_groups_by_component(
    #     features_df,
    #     base_output_folder=output_root
    # )

    # viz_feature_values_vs_importance_by_component(
    #     features_df,
    #     base_output_folder=output_root,
    # )

    if explanations_path:
        explanations_path = Path(explanations_path) / "explanations.json"
        # plot_audiolime_predictions_influence_features(
        #     features_df, Path(explanations_path), output_root
        # )

        # plot_audiolime_3rows_multicolumn(
        #     features_df=features_df,
        #     lime_json_path=Path(explanations_path),
        #     outputdir=output_root
        # )

        plot_feature_correlation_r_heatmaps(
            features_df=features_df,
            lime_json_path=Path(explanations_path),
            comp_version=comp_version,
            outputdir=output_root,
        )
        
        plot_feature_comparison_table(
            features_df=features_df,
            lime_json_path=Path(explanations_path),
            comp_version=comp_version,
            outputdir=output_root,
        )

if __name__ == "__main__":
    main()