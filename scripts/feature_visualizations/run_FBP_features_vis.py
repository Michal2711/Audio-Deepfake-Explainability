import json
from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import re

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
        description="Visualize FBP feature importance results"
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
                            result[col_name] = val
                    
                    return result

def load_and_prepare_data_full(json_file):
    """
    Load JSON data and preserve ALL sub-features from nested structure.
    
    Data structure example:
    {
        model_name: {
            track_id: {
                "type": "band",
                "bands": {
                    band_key: {
                        "features": {
                            "duration": 120.0,
                            "rms_wave": {"min": ..., "mean": ..., "std": ..., "max": ...},
                            "jitter": {"jitter_local": ..., "jitter_rap": ..., ...},
                            ...
                        },
                        "band_meta": {
                            "component": "mixture" | "vocals0" | "drums0" | "bass0" | "other0",
                            "rank": 1,
                            "importance": -0.1234,
                            "abs_importance": 0.1234,
                            "tstart": 0,
                            "tend": 512,
                            "fstart": 0,
                            "fend": 512,
                            "start_time_sec": 0.0,
                            "end_time_sec": 5.944308390022676,
                            "band_type": NEGATIVE | POSITIVE,
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
    - DataFrame with collumns: model, track, band_key, component, band_type, data_type, [all_features}]
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
            
            if not isinstance(track_data, dict) or 'bands' not in track_data:
                continue
            
            bands_root = track_data.get('bands', {})
            track_type = track_data.get('type', 'unknown')

            for band_key, band_data in bands_root.items():

                if not isinstance(band_data, dict) or 'features' not in band_data:
                    continue

                features = band_data.get('features', {})
                band_meta = band_data.get('band_meta', {})
                            
                row = {
                    'model': model_name,
                    'track': track_key,
                    'band_key': band_key,
                    'data_type': type_mapping.get(model_name, model_name),
                    **flatten_feature(band_meta)
                }
                
                flattened = flatten_feature(features)
                row.update(flattened)
                
                all_rows.append(row)
    

    features_df = pd.DataFrame(all_rows)
    features_df['band_key'] = features_df['band_key'].str.replace('mixture_', '').str.replace('_', '-').str.replace('.0', '')
    
    if features_df.empty:
        print("⚠️ Warning: No data loaded from JSON file!")
        return features_df, []
    
    exclude_cols = {'model', 'track', 'band_key', 'data_type'}
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    print(f"\n{'='*80}")
    print(f"✅ Data loaded successfully!")
    print(f"   • Models: {features_df['model'].unique().tolist()}")
    print(f"   • Total records: {len(features_df)}")
    print(f"   • Total features: {len(feature_cols)}")
    print(f"   • Sample features: {feature_cols[:10]}")
    print(f"{'='*80}\n")
    
    return features_df, feature_cols

def load_fbp_bands_explanations(root_folder: Path) -> pd.DataFrame:
    all_rows = []
    typemapping = {'ElevenLabs': 'GENERATED', 'REAL': 'REAL', 'SUNO': 'GENERATED',
                   'SUNOPRO': 'GENERATED', 'UDIO': 'GENERATED'}
    
    fbp_results_path = root_folder / "fbp_results.json"
    predictions_dict = {}
    if fbp_results_path.exists():
        with open(fbp_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for model_name, tracks_dict in data.items():
            for track_name, track_data in tracks_dict.items():
                for comp_name, comp_data in track_data.get("components", {}).items():
                    pred_score = comp_data.get('baseline_pred_mean', np.nan)
                key = f"{model_name}_{track_name}_{comp_name}"
                predictions_dict[key] = float(pred_score)
        print(f"Loaded {len(predictions_dict)} predictions")
    else:
        print(f" fbp_results.json w {root_folder}")
    
    bands_folder = Path(root_folder / "bands")

    for model_folder in bands_folder.iterdir():
        if not model_folder.is_dir() or model_folder.name == 'fbp_results.json': continue
        model_name = model_folder.name
        
        for track_folder in model_folder.iterdir():
            if not track_folder.is_dir(): continue
            track_name = track_folder.name
                        
            for comp_folder in track_folder.iterdir():
                if not comp_folder.is_dir(): continue
                comp_name = comp_folder.name

                pred_key = f"{model_name}_{track_name}_{comp_name}"
                pred_score = predictions_dict.get(pred_key, np.nan)
                
                json_pattern = f"{track_name}_bands_metadata.json"
                json_file = comp_folder / json_pattern
                if not json_file.exists(): continue
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    bands = data.get('bands', [])
                    for band in bands:
                        row = {
                            'model': model_name,
                            'track': track_name,
                            'component': band.get('component', comp_name),
                            'band_key': f"{int(band.get('low', 0))}-{int(band.get('high', 0))}Hz",  # 20-100Hz
                            'low': float(band.get('low', 0)),
                            'high': float(band.get('high', 0)),
                            'importance': float(band.get('importance', 0)),
                            'abs_importance': float(band.get('abs_importance', 0)),
                            'type': band.get('type', 'UNKNOWN'),
                            'prediction_score': pred_score
                        }
                        all_rows.append(row)
                        
                except Exception as e:
                    print(f"Error {json_file}: {e}")
    
    fbp_df = pd.DataFrame(all_rows)
    if fbp_df.empty:
        print("Warning: No FBP data found")
    
    return fbp_df

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

def add_freq_band_from_band_key(features_df):
    df = features_df.copy()
    df['band_key'] = df['band_key'].astype(str)
    
    conditions = [
        # default FBP bands
        df['band_key'].str.contains('20.0_100.0Hz', case=False, na=False),
        df['band_key'].str.contains('100.0_250.0Hz', case=False, na=False),
        df['band_key'].str.contains('250.0_2000.0Hz', case=False, na=False),
        df['band_key'].str.contains('2000.0_4000.0Hz', case=False, na=False),
        df['band_key'].str.contains('4000.0_8000.0Hz', case=False, na=False),
        df['band_key'].str.contains('8000.0_16000.0Hz', case=False, na=False),
        # detailed voice preset bands
        df['band_key'].str.contains('20.0_60.0Hz', case=False, na=False),
        df['band_key'].str.contains('60.0_250.0Hz', case=False, na=False),
        df['band_key'].str.contains('250.0_500.0Hz', case=False, na=False),
        df['band_key'].str.contains('500.0_2000.0Hz', case=False, na=False),
        df['band_key'].str.contains('2000.0_4000.0Hz', case=False, na=False),
        df['band_key'].str.contains('4000.0_6000.0Hz', case=False, na=False),
        df['band_key'].str.contains('6000.0_12000.0Hz', case=False, na=False),
        df['band_key'].str.contains('12000.0_21000.0Hz', case=False, na=False),
        # high resolution preset bands
        df['band_key'].str.contains('20.0_60.0Hz', case=False, na=False),
        df['band_key'].str.contains('60.0_100.0Hz', case=False, na=False),
        df['band_key'].str.contains('100.0_250.0Hz', case=False, na=False),
        df['band_key'].str.contains('250.0_500.0Hz', case=False, na=False),
        df['band_key'].str.contains('500.0_1000.0Hz', case=False, na=False),
        df['band_key'].str.contains('1000.0_2000.0Hz', case=False, na=False),
        df['band_key'].str.contains('2000.0_4000.0Hz', case=False, na=False),
        df['band_key'].str.contains('4000.0_6000.0Hz', case=False, na=False),
        df['band_key'].str.contains('6000.0_8000.0Hz', case=False, na=False),
        df['band_key'].str.contains('8000.0_10000.0Hz', case=False, na=False),
        df['band_key'].str.contains('10000.0_12000.0Hz', case=False, na=False),
        df['band_key'].str.contains('12000.0_16000.0Hz', case=False, na=False),
        df['band_key'].str.contains('16000.0_21000.0Hz', case=False, na=False),
    ]
    
    choices = [
        # default FBP bands
        '20-100 Hz',
        '100-250 Hz',
        '250-2000 Hz',
        '2000-4000 Hz',
        '4000-8000 Hz',
        '8000-16000 Hz',
        # detailed voice preset bands
        '20-60 Hz',
        '60-250 Hz',
        '250-500 Hz',
        '500-2000 Hz',
        '2000-4000 Hz',
        '4000-6000 Hz',
        '6000-12000 Hz',
        '12000-21000 Hz',
        # high resolution preset bands
        '20-60 Hz',
        '60-100 Hz',
        '100-250 Hz',
        '250-500 Hz',
        '500-1000 Hz',
        '1000-2000 Hz',
        '2000-4000 Hz',
        '4000-6000 Hz',
        '6000-8000 Hz',
        '8000-10000 Hz',
        '10000-12000 Hz',
        '12000-16000 Hz',
        '16000-21000 Hz'
    ]
    
    df['freq_band'] = np.select(conditions, choices, default='other')
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

def safe_reindex_fill(df, index_col, target_idx):
    df_sorted = df.sort_values(index_col)
    df_reidx = df_sorted.set_index(index_col).reindex(target_idx, method='ffill')
    df_final = df_reidx.ffill().reset_index()
    df_final.rename(columns={'index': index_col}, inplace=True)
    return df_final

def plot_fbp_predictions_influence_features(
    features_df: pd.DataFrame, 
    fbp_json_path: Path, 
    output_dir: Path
):
    fbp_explanations_df = load_fbp_bands_explanations(fbp_json_path)

    setup_professional_style()
    sns.set_theme(style="whitegrid")

    # Merge features + FBP explanations
    merge_cols = ['model', 'track', 'component', 'band_key']
    full_df = pd.merge(
        features_df,
        fbp_explanations_df, 
        on=merge_cols, 
        how='inner'
    )
    
    full_df['importance'] = full_df['importance_y']
    output_dir = Path(output_dir) / "fbp_3rows_per_band"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exclude_cols = ['model', 'track', 'component', 'data_type', 
                    'band_key', 'band_type', 'prediction_score', 
                    'importance', 'abs_importance', 'low_freq', 
                    'high_freq', 'track_stem'
    ]
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    
    models = sorted(full_df['model'].unique())
    
    for model in models:
        model_df = full_df[full_df['model'] == model]
        model_dir = output_dir / model.replace('/', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        unique_bands = sorted(model_df['band_key'].unique())
        print(f"Model {model}: {len(unique_bands)} bands")
        
        for band_key in unique_bands:
            band_df = model_df[model_df['band_key'] == band_key].reset_index(drop=True)
            if len(band_df) == 0: 
                print(f"  Skip {model}/{band_key}: No data")
                continue
            
            band_df['file_index'] = range(len(band_df))
            if 'file_index' not in band_df.columns:
                band_df = band_df.sort_values('file_index').groupby('file_index').first().reset_index()
            n_files = len(band_df)
            track_stems = band_df['track'].tolist()
            
            band_features = [c for c in feature_cols if band_df[c].notna().sum() > 0]
            print(f"  Band {band_key}: {len(band_features)} features")
            
            for feat_col in band_features:

                base_feat_name = re.sub(r'_(mean|std|min|max)$', '', feat_col)

                d_pred = band_df[[
                    'file_index', 'prediction_score'
                ]].dropna()
                d_fbp = band_df[[
                    'file_index', 'importance'
                ]].dropna()
                d_feat = band_df[[
                    'file_index', feat_col
                ]].dropna()

                all_idx = range(n_files)
                d_pred = safe_reindex_fill(d_pred, 'file_index', all_idx)
                d_fbp = safe_reindex_fill(d_fbp, 'file_index', all_idx)
                d_feat = safe_reindex_fill(d_feat, 'file_index', all_idx)

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, height_ratios=[1, 1, 1])
                
                # Rząd 1: Predictions
                ax1.plot(d_pred['file_index'], d_pred['prediction_score'], 'o-', linewidth=3, markersize=8, 
                        color='darkred', alpha=0.9, label='Predictions (P>0.5=FAKE)')
                ax1.axhline(y=0.5, color='black', ls='-', lw=2.5, alpha=0.8)
                ax1.set_ylabel('Predictions', fontweight='bold', fontsize=12)
                ax1.grid(alpha=0.3, ls='--')
                ax1.legend(loc='upper right')
                
                # Rząd 2: FBP Influence
                ax2.plot(d_fbp['file_index'], d_fbp['importance'], 'o-', linewidth=3, markersize=8, 
                        color='purple', alpha=0.85, label=f'FBP {band_key}')
                ax2.axhline(y=0, color='gray', ls=':', lw=2)
                ax2.set_ylabel('FBP Importance', fontweight='bold', fontsize=12)
                ax2.grid(alpha=0.3, ls='--')
                ax2.legend(loc='upper right')
                
                # Rząd 3: Physical Feature
                ax3.plot(d_feat['file_index'], d_feat[feat_col], 'o-', linewidth=3, markersize=8, 
                        color='steelblue', alpha=0.85, label=feat_col.replace('_', ' ').title())
                ax3.set_xlabel('File Index')
                ax3.set_ylabel('Physical Feature', fontweight='bold', fontsize=12)
                ax3.grid(alpha=0.3, ls='--')
                ax3.legend(loc='upper right')
                
                ax1.set_title(f'{model} | {band_key}', fontsize=14, fontweight='bold', pad=20)
                
                plt.tight_layout()
                
                short_labels = [
                    f'{i}: {stem[:25]}'
                    for i, stem in enumerate(track_stems)
                ]
                fig.text(
                    1.02, 
                    0.48, 
                    'File Mapping:\n' + '\n'.join(short_labels),
                    fontsize=15, 
                    va='center', 
                    ha='left',
                    bbox=dict(
                        facecolor='white', 
                        edgecolor='gray', 
                        boxstyle='round,pad=0.3',
                        alpha=0.95
                    )
                )
                
                safe_band = re.sub(r'[^a-zA-Z0-9]', '_', band_key)
                safe_feat = re.sub(r'[^a-zA-Z0-9]', '_', feat_col)
                base_feat_dir = model_dir / safe_band / base_feat_name
                base_feat_dir.mkdir(exist_ok=True, parents=True)
                outfile = base_feat_dir / f'{safe_band}_{safe_feat}_3rows.png'

                plt.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"  Saved: {safe_band}_{safe_feat}_3rows.png")
    
def plot_fbp_3rows_multicolumn(
    features_df: pd.DataFrame,
    fbp_json_path: Path,
    output_dir: Path
):
    setup_professional_style()
    sns.set_theme(style="whitegrid")

    fbp_explanations_df = load_fbp_bands_explanations(fbp_json_path)

    merge_cols = ['model', 'track', 'component', 'band_key']
    full_df = pd.merge(
        features_df,
        fbp_explanations_df,
        on=merge_cols,
        how='inner',
        suffixes=('_feat', '_fbp')
    )
    if full_df.empty:
        print("Empty merge result")
        return

    exclude_cols = [
        'model', 'track', 'band_key', 'data_type',
        'component', 'importance', 'abs_importance',
        'low_freq', 'high_freq', 'band_type', 'track_stem',
    ]
    feature_cols = []
    for c in features_df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(features_df[c]):
            feature_cols.append(c)


    out_root = Path(output_dir) / "fbp_3rows_multicolumn"
    out_root.mkdir(parents=True, exist_ok=True)

    models = sorted(full_df['model'].unique())

    for model in models:
        model_df = full_df[full_df['model'] == model].copy()
        model_dir = out_root / model.replace('/', '_')
        model_dir.mkdir(parents=True, exist_ok=True)

        track_stems = sorted(model_df['track'].unique())
        nfiles = len(track_stems)
        if nfiles == 0:
            continue

        if 'prediction_score' in model_df.columns:
            pred_col = 'prediction_score'
        elif 'prediction_score' in model_df.columns:
            pred_col = 'prediction_score'
        else:
            pred_col = None

        if pred_col is not None:
            model_pred = model_df.groupby('track')[pred_col].mean()
            pred_values = [model_pred.get(t, np.nan) for t in track_stems]
        else:
            pred_values = [0.5] * nfiles

        bands = sorted(model_df['band_key'].unique())
        n_comp = len(bands)
        if n_comp == 0:
            continue

        for feat_col in feature_cols:
            if model_df[feat_col].isna().all():
                continue

            base_feat_name = re.sub(r'_(mean|std|min|max)$', '', feat_col)
            base_feat_dir = model_dir / base_feat_name
            base_feat_dir.mkdir(exist_ok=True, parents=True)

            fig, axes = plt.subplots(
                3, n_comp,
                figsize=(5 * n_comp, 12),
                sharex='col',
                sharey='row'
            )
            if n_comp == 1:
                axes = axes.reshape(3, 1)

            for j in range(n_comp):
                ax = axes[0, j]
                ax.plot(
                    range(nfiles),
                    pred_values,
                    'o-',
                    lw=2.5,
                    ms=6,
                    color='darkred',
                    alpha=0.9,
                    label='Predictions'
                )
                ax.axhline(0.5, color='black', ls='-', lw=2, alpha=0.8)
                if j == 0:
                    ax.set_ylabel('Predictions\n(P>0.5=FAKE)', fontweight='bold')
                ax.set_title(bands[j], fontweight='bold')
                ax.grid(alpha=0.3, ls='--')
            axes[0, 0].legend(loc='upper left')

            for j, band in enumerate(bands):
                ax = axes[1, j]
                band_data = model_df[model_df['band_key'] == band]

                if 'importance_fbp' in band_data.columns:
                    imp_series = band_data.groupby('track')['importance_fbp'].mean()
                elif 'importance' in band_data.columns:
                    imp_series = band_data.groupby('track')['importance'].mean()
                else:
                    imp_series = pd.Series(dtype=float)

                imp_vals = [imp_series.get(t, np.nan) for t in track_stems]

                ax.plot(
                    range(nfiles),
                    imp_vals,
                    'o-',
                    lw=2.5,
                    ms=6,
                    color='purple',
                    alpha=0.85,
                    label=f'FBP {band}'
                )
                ax.axhline(0, color='gray', ls=':', lw=2)
                if j == 0:
                    ax.set_ylabel('FBP Importance', fontweight='bold')
                ax.grid(alpha=0.3, ls='--')

            axes[1, 0].legend(loc='upper left')

            for j, band in enumerate(bands):
                ax = axes[2, j]
                band_data = model_df[model_df['band_key'] == band]

                feat_series = band_data.groupby('track')[feat_col].mean()
                feat_vals = [feat_series.get(t, np.nan) for t in track_stems]

                ax.plot(
                    range(nfiles),
                    feat_vals,
                    'o-',
                    lw=2.5,
                    ms=6,
                    color='steelblue',
                    alpha=0.85
                )
                if j == 0:
                    ax.set_ylabel(
                        feat_col.replace('_', ' ').title(),
                        fontweight='bold'
                    )
                ax.set_xlabel('Track index')
                ax.grid(alpha=0.3, ls='--')

            plt.suptitle(
                f'{model} | {feat_col.replace("_", " ").title()} – all bands',
                fontsize=16,
                y=0.98,
                fontweight='bold'
            )
            plt.tight_layout()

            short_labels = [
                f'{i}: {stem[:25]}'
                for i, stem in enumerate(track_stems)
            ]
            fig.text(
                1.02,
                0.48,
                'File Mapping:\n' + '\n'.join(short_labels),
                fontsize=15,
                va='center',
                ha='left',
                bbox=dict(
                    facecolor='white',
                    edgecolor='gray',
                    boxstyle='round,pad=0.3',
                    alpha=0.95
                )
            )

            safe_feat = re.sub(r'[^a-zA-Z0-9]', '_', feat_col)
            out_file = base_feat_dir / f'{safe_feat}_allbands_3rows.png'
            plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"[FBP multicol] Saved {out_file}")

def viz_component_pos_neg_boxplots(
    features_df,
    base_output_folder=Path('./'),
    component_col='component',
    importance_col='importance',
    sign_col='influence_sign'
):
    setup_professional_style()
    
    base_folder = Path(f'{base_output_folder}/visualizations_boxplot_component_pos_neg')
    base_folder.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("Creating COMPONENT-SPLIT boxplot visualizations (positive vs negative)...")
    print(f"{'='*80}\n")
    
    if component_col not in features_df.columns:
        raise ValueError(f"Column '{component_col}' not found in features_df")
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
        'model', 'track', 'band_key', 'data_type',
        component_col, importance_col, 'influence_sign'
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
    
    components = sorted(df[component_col].dropna().unique())
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
    print(f"{'='*80}\n")

def viz_feature_groups_by_freq_band(
    features_df,
    base_output_folder=Path('./'),
    importance_col='importance',
    band_type_col='band_type',
    freq_band_col='freq_band'
):
    setup_professional_style()

    df = features_df.copy()

    if band_type_col not in df.columns:
        df['influence_sign'] = np.where(df[importance_col] >= 0, 'positive', 'negative')
    else:
        df['influence_sign'] = df[band_type_col].astype(str).str.lower()

    if freq_band_col not in df.columns:
        raise ValueError(f"Column '{freq_band_col}' not found")

    bands = sorted(df[freq_band_col].dropna().unique())

    print(f"Processing {len(bands)} frequency bands...")

    for band in bands:
        band_df = df[df[freq_band_col] == band].copy()
        if band_df.empty:
            print(f"⚠️ Skipping empty band: {band}")
            continue

        band_base_folder = Path(base_output_folder) / 'by_freq_band_feature_sign' / band.replace(' ', '_')
        band_base_folder.mkdir(parents=True, exist_ok=True)

        models = sorted(band_df['model'].dropna().unique())
        signs = ['negative', 'positive']

        exclude_cols = {
            'model', 'track', 'band_key', 'data_type',
            importance_col, 'influence_sign', freq_band_col
        }
        band_meta_cols = {
            'component', 'importance', 'abs_importance',
            'low_freq', 'high_freq', 'band_type', 'model', 'track_stem'
        }

        all_cols = [
            col for col in band_df.columns
            if (
                col not in exclude_cols and
                col not in band_meta_cols and
                band_df[col].dtype in ['float64', 'float32', 'int64', 'int32'] and
                band_df[col].notna().sum() > 0
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

        print(f"  {band}: {len(feature_groups)} feature groups")

        for feature_base, columns_list in sorted(feature_groups.items()):
            feature_folder = band_base_folder / feature_base
            feature_folder.mkdir(parents=True, exist_ok=True)

            if len(columns_list) == 1 and columns_list[0][1] == 'single':
                col = columns_list[0][0]
                _viz_single_feature_in_band(band_df, col, feature_folder, models, signs, feature_base)
            else:
                stat_order = ['min', 'mean', 'std', 'max']
                columns_sorted = sorted(
                    columns_list,
                    key=lambda x: next((i for i, stat in enumerate(stat_order) if stat == x[1]), 999)
                )

                for col, stat in columns_sorted:
                    stat_label = stat.upper() if stat != 'single' else col
                    _viz_single_feature_in_band(band_df, col, feature_folder, models, signs, feature_base, stat_label)

        print(f"  ✅ {band_base_folder} done")

def _viz_single_feature_in_band(band_df, col, feature_folder, models, signs, feature_base, stat_label=None):
    feat_df = band_df[band_df[col].notna()].copy()
    if feat_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax_models, ax_global = axes
    sign_col = 'influence_sign'

    # ===== Left panel: per model =====
    plot_data = []
    x_labels = []

    for model in models:
        for s in signs:
            mask = (feat_df['model'] == model) & (feat_df[sign_col] == s)
            data = feat_df.loc[mask, col].dropna()
            if len(data) > 0:
                plot_data.append(data.values)
                x_labels.append(f'{model}\n{s}')
            else:
                plot_data.append(None)
                x_labels.append(f'{model}\n{s}')

    non_empty_indices = [i for i, d in enumerate(plot_data) if d is not None and len(d) > 0]
    if not non_empty_indices:
        print(f" ⚠️ SKIPPED: No valid data for {col}")
        plt.close(fig)
        return

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

    title_suffix = f' – {stat_label}' if stat_label else ''
    ax_models.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax_models.set_title(
        f'{col.replace("_", " ")}{title_suffix} per model\n'
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

    # ===== Right panel: global =====
    global_plot_data = []
    global_labels = []

    for s in signs:
        data = feat_df.loc[feat_df[sign_col] == s, col].dropna()
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
            f'{col.replace("_", " ")}{title_suffix}\n'
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
        f'{feature_base.replace("_", " ").title()} – {stat_label or col}',
        fontsize=16, fontweight='bold', y=0.97
    )

    plt.tight_layout(rect=[0.03, 0.14, 0.97, 0.93])

    stats_text = format_influence_statistics_box(x_labels, plot_data)
    add_bottom_stats_panel(fig, ax_models, stats_text, width_frac=0.45, y_margin=0.04)

    global_stats_text = format_influence_statistics_box(global_labels, global_plot_data)
    add_bottom_stats_panel(fig, ax_global, global_stats_text, width_frac=0.30, y_margin=0.04)

    out_file = feature_folder / f'{col}_{stat_label or "single"}_most_influential_pos_neg.png'
    plt.savefig(
        out_file, dpi=300, bbox_inches='tight',
        facecolor='white', edgecolor='none'
    )
    plt.close(fig)
    print(f" ✓ Saved: {out_file}")

def viz_single_feature_vs_importance_in_freq_band(
    band_df,
    feature_col,
    feature_folder,
    feature_label,
    importance_col="importance",
    sign_col="influence_sign",
):
    required_cols = [feature_col, importance_col, "model", sign_col]
    for col in required_cols:
        if col not in band_df.columns:
            return

    feature_df = band_df[required_cols].dropna(
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

        color_hex = PROFESSIONAL_COLORS.get(model_name, "#333333")

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
    fig.supylabel("Band importance", fontsize=13, fontweight="bold")

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

def viz_feature_values_vs_importance_by_freq_band(
    features_df,
    base_output_folder=Path("."),
    importance_col="importance",
    freq_band_col="freq_band",
    band_type_col="band_type",
):
    setup_professional_style()

    df = features_df.copy()

    if importance_col not in df.columns:
        raise ValueError(f"Column '{importance_col}' not found")

    if band_type_col in df.columns:
        df["influence_sign"] = (
            df[band_type_col].astype(str).str.lower().replace({"positive": "positive", "negative": "negative"})
        )
    else:
        df["influence_sign"] = np.where(
            df[importance_col] >= 0, "positive", "negative"
        )

    if freq_band_col not in df.columns:
        raise ValueError(f"Column '{freq_band_col}' not found (run add_freq_band_from_band_key first).")

    bands = sorted(df[freq_band_col].dropna().unique().tolist())
    print(f"Processing {len(bands)} frequency bands for feature vs importance...")

    for band in bands:
        band_df = df[df[freq_band_col] == band].copy()
        if band_df.empty:
            print(f"Skipping empty band {band}")
            continue

        band_safe = str(band).replace(" ", "_")
        band_base_folder = (
            Path(base_output_folder)
            / "by_freq_band_feature_vs_importance"
            / band_safe
        )
        band_base_folder.mkdir(parents=True, exist_ok=True)

        exclude_cols = [
            "model",
            "track",
            "band_key",
            "data_type",
            importance_col,
            "influence_sign",
            freq_band_col,
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

        all_cols = [
            col
            for col in band_df.columns
            if col not in exclude_cols
            and col not in band_meta_cols
            and band_df[col].dtype in (np.float64, np.float32, np.int64, np.int32)
            and band_df[col].notna().sum() > 0
        ]

        feature_groups = defaultdict(list)
        for col in all_cols:
            parts = col.split("_")
            if len(parts) > 1 and parts[-1] in ("min", "mean", "std", "max"):
                base_name = "_".join(parts[:-1])
                stat = parts[-1]
            else:
                base_name = col
                stat = "single"
            feature_groups[base_name].append((col, stat))

        print(f"{band}: {len(feature_groups)} feature groups")

        for feature_base, columns_list in sorted(feature_groups.items()):
            feature_folder = band_base_folder / feature_base
            feature_folder.mkdir(parents=True, exist_ok=True)

            if len(columns_list) == 1 and columns_list[0][1] == "single":
                col, _ = columns_list[0]
                feature_label = f"{feature_base}"
                viz_single_feature_vs_importance_in_freq_band(
                    band_df,
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
                    viz_single_feature_vs_importance_in_freq_band(
                        band_df,
                        col,
                        feature_folder,
                        feature_label,
                        importance_col=importance_col,
                        sign_col="influence_sign",
                    )

        print(f"{band_base_folder} done")


def main():
    args = parse_args()
    config = load_yaml(Path(args.config))

    data_cfg = config.get("data", {})
    explanations_cfg = config.get("explanations_data", {})
    output_cfg = config.get("output", {})
    fbp_band_features_cfg = config.get("fbp_band_features", {})
    band_version = fbp_band_features_cfg.get("version", "separated")

    data_root = Path(data_cfg.get("features_path"))
    explanations_path = explanations_cfg.get("explanations_path")
    result_root = Path(output_cfg.get("result_path"))

    data_root = data_root / "separated_bands" if band_version == "separated" else data_root / "reversed_separated_bands"
    features_path = data_root / "fbp_bands"  / "fbp_band_features.json"
    
    result_root = result_root / "separated_bands" if band_version == "separated" else result_root / "reversed_separated_bands"
    
    output_root = result_root / "features_visualization"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Output root: {output_root}")
    print("Visualizing FBP feature importance results")
    print("=" * 70)

    features_df, features_to_analyze = load_and_prepare_data_full(features_path)

    print(f"\n✓ Data loaded: {len(features_df)} samples, {len(features_to_analyze)} features")
    print(f"✓ Models: {features_df['model'].value_counts().to_dict()}\n")

    features_df = add_freq_band_from_band_key(features_df)

    # viz_component_pos_neg_boxplots(
    #     features_df,
    #     base_output_folder=output_root,
    # )

    # viz_feature_groups_by_freq_band(
    #     features_df,
    #     base_output_folder=output_root
    # )

    # viz_feature_values_vs_importance_by_freq_band(
    #     features_df,
    #     base_output_folder=output_root,
    # )

    plot_fbp_predictions_influence_features(
        features_df=features_df, 
        fbp_json_path=Path(explanations_path), 
        output_dir=output_root
    )

    plot_fbp_3rows_multicolumn(
        features_df=features_df,
        fbp_json_path=Path(explanations_path),
        output_dir=output_root
    )

if __name__ == "__main__":
    main()