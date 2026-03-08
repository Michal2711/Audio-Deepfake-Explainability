import json
import argparse
from pathlib import Path
from typing import Sequence, Dict, Any
import re
import yaml
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    
PALETTE = [
    '#1f77b4',
    '#ff7f0e',  
    '#2ca02c',  
    '#d62728',  
    '#9467bd',  
    '#8c564b',  
    '#e377c2',  
    '#7f7f7f',  
    '#bcbd22',  
    '#17becf',  
    '#ff9896',  
    '#98df8a',  
    '#c5b0d5',  
    '#c49c94',  
    '#f7b6d2'
]

def try_num(s):
    if isinstance(s, bytes):
        s = s.decode('utf-8', errors='ignore')
    match = re.match(r'^(\d+)', s)
    return int(match.group(1)) if match else 999999


def extract_run_label(file_path: str) -> str:
    path = Path(file_path)
    name = str(path).lower()
    
    if 'minus14' not in name and 'minus23' not in name :
        return 'Original'
    elif 'base' in name and 'minus14' in name:
        return 'm14_base'
    elif 'base' in name and 'minus23' in name:
        return 'm23_base'
    elif 'mp3_192' in name and 'minus14' in name:
        return 'm14_mp3_192'
    elif 'mp3_192' in name and 'minus23' in name:
        return 'm23_mp3_192'
    elif 'noise_snr30' in name and 'minus14' in name:
        return 'm14_noise_snr30'
    elif 'noise_snr30' in name and 'minus23' in name:
        return 'm23_noise_snr30'
    elif 'resample22k' in name and 'minus14' in name:
        return 'm14_resample_22k'
    elif 'resample22k' in name and 'minus23' in name:
        return 'm23_resample22k'
    elif 'reverb_room' in name and 'minus14' in name:
        return 'm14_reverb_room'
    elif 'reverb_room' in name and 'minus23' in name:
        return 'm23_reverb_room'
    else:
        return path.parent.name if path.parent.name != '.' else path.stem[:20]

def extract_data_source(text: str) -> str:
    text_upper = text.upper()
    if 'ELEVENLABS' in text_upper:
        return 'ElevenLabs'
    elif 'REAL' in text_upper or 'MP3' in text_upper:
        return 'REAL'
    elif 'SUNO' in text_upper:
        return 'SUNO'
    elif 'UDIO' in text_upper:
        return 'UDIO'
    else:
        return 'Other'

def to_long_frame(data: Dict[str, Any], run_label: str) -> pd.DataFrame:
    rows = []
    DEFAULT_COMPONENTS = ["vocals0", "piano0", "drums0", "bass0", "other0"]
    
    for model_name, items in data.items():
        if not isinstance(items, dict):
            continue
        for file_name, results in items.items():
            explanations = results.get('explanations', {})
            file_path = explanations.get('file_path', file_name)
            comp_influences = explanations.get('component_influences', {})
            
            comp_items = comp_influences.items() if comp_influences else [(c, np.nan) for c in DEFAULT_COMPONENTS]
            
            for comp, value in comp_items:
                rows.append({
                    "data_source": model_name,
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_index": int(results.get('track_id', 0)),
                    "component": comp,
                    "value": float(value) if not pd.isna(value) else np.nan,
                    "run": run_label
                })
    return pd.DataFrame(rows)

def load_audio_lime_explanations(file_paths: Sequence[str]) -> pd.DataFrame:
    dfs = []
    runs_labels = ''
    
    for p in file_paths:
        run_label = extract_run_label(p)
        runs_labels += f"{run_label}_"
        
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df_run = to_long_frame(data, run_label)
        dfs.append(df_run)
        print(f"✅ Loaded {len(df_run)} rows from {p} (run: {run_label})")
    
    df_all = pd.concat(dfs, ignore_index=True)
    
    keys_per_run = [df.groupby(['data_source', 'file_name', 'component']).size() > 0 
                    for _, df in df_all.groupby('run')]
    
    common_mask = keys_per_run[0].reindex(keys_per_run[0].index).fillna(False)
    for mask in keys_per_run[1:]:
        common_mask &= mask.reindex(common_mask.index).fillna(False)
    
    df_common = df_all[df_all.set_index(['data_source', 'file_name', 'component']).index.isin(
        common_mask[common_mask].index)].copy()
    
    df_common = df_common.sort_values(["data_source", "component", "file_index", "run"]).reset_index(drop=True)
    
    print(f"✅ Common data: {len(df_common)} rows across {df_common['data_source'].nunique()} sources")
    print(f"   Runs: {sorted(df_common['run'].unique())}")
    
    return df_common, runs_labels.strip('_')

def plot_audio_lime_influences(df_common: pd.DataFrame, output_dir: Path = None):
    sns.set_theme(style="whitegrid")
    
    providers = sorted(df_common["data_source"].unique())
    components_order = ["vocals0", "piano0", "drums0", "bass0", "other0"]
    
    run_names = sorted(df_common['run'].unique())
    legend_runs = " vs ".join(run_names)
    
    for prov in providers:
        dprov = df_common[df_common["data_source"] == prov].copy()
        if dprov.empty:
            continue
        
        comps = [c for c in components_order if c in dprov["component"].unique()]
        if not comps:
            continue
        
        g = sns.FacetGrid(
            dprov[dprov["component"].isin(comps)],
            col="component",
            col_order=comps,
            hue="run",
            height=3.2,
            aspect=1.2,
            sharey=False,
            palette='husl'
        )
        g.map_dataframe(sns.lineplot, x="file_index", y="value")
        g.add_legend(title="Run")
        g.set_axis_labels("file index", "influence")
        g.set_titles(col_template="{col_name}")
        g.fig.suptitle(f"{prov}: AudioLIME influence vs file index ({legend_runs})", y=1.05, fontsize=12)
        
        if output_dir:
            outfile = output_dir / f"{prov}_audiolime_influences.png"
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {outfile}")
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="AudioLIME runs results comparison - FacetGrid style")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(Path(args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    df_common, runs_labels = load_audio_lime_explanations(config.get('files', []))
    
    output_cfg = config.get('output', {})
    output_dir = Path(output_cfg.get('result_path', 'results/AudioLIME/Runs_comparison')) / runs_labels
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_audio_lime_influences(df_common, output_dir)
    print(f"✅ All plots saved to {output_dir}")

if __name__ == "__main__":
    main()
