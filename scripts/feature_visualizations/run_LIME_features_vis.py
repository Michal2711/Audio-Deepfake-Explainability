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
                outfile = comp_dir / f"{safe_comp}_{safe_feat}_3rows.png"
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
            outfile = model_dir / f"{safe_feat}_all_components.png"
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

    viz_component_pos_neg_boxplots(
        features_df,
        base_output_folder=output_root,
    )

    viz_feature_groups_by_component(
        features_df,
        base_output_folder=output_root
    )

    viz_feature_values_vs_importance_by_component(
        features_df,
        base_output_folder=output_root,
    )

    if explanations_path:
        explanations_path = Path(explanations_path) / "explanations.json"
        plot_audiolime_predictions_influence_features(
            features_df, Path(explanations_path), output_root
        )

        plot_audiolime_3rows_multicolumn(
            features_df=features_df,
            lime_json_path=Path(explanations_path),
            outputdir=output_root
        )

if __name__ == "__main__":
    main()