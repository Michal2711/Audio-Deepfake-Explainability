# scripts/compare_predictions_across_runs.py
import json
import argparse
from pathlib import Path
from typing import Sequence, List
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

def load_runs_with_mod(file_paths: Sequence[str], threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    runs_labels = ''
    for p in file_paths:
        run_label = extract_run_label(p)
        runs_labels += f"{run_label}_"
        
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
                
        for model_name, audio_items in data.items():
            for audio_stem, rec in audio_items.items():
                pred = rec.get('prediction', np.nan)
                if np.isnan(pred):
                    continue
                
                rows.append({
                    'run_path': str(p),
                    'run': run_label,
                    'source': model_name,
                    'idx': audio_stem,
                    'score_fake_prob': pred,
                })
    
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid data loaded!")
    
    df['pred_by_threshold'] = (df['score_fake_prob'] >= threshold).map({True: 'Fake', False: 'Real'})
    
    df['idx_sort'] = df['idx'].map(try_num)

    print(f"✅ Loaded {len(df)} predictions from {len(file_paths)} files")
    print(f"   Runs: {sorted(df['run'].unique())}")
    print(f"   Models: {sorted(df['source'].unique())}")
    return df, runs_labels.strip('_')

def plot_with_right_box(df: pd.DataFrame, source: str, idxs: List, idx_pos: dict, 
                       PALETTE: List, runs: List, short_labels: List, 
                       annotate_decision: bool = False, figsize=(14, 6),
                       output_dir: Path = None):
    setup_professional_style()
    fig, ax = plt.subplots(figsize=figsize)
    markers = ['o','s','D','^','v','P','X','*','h','8']
    
    for r_i, run in enumerate(runs):
        g = df[(df['run'] == run) & (df['source'] == source)].sort_values('idx_sort')
        if len(g) == 0:
            continue
            
        x = [idx_pos[v] + (r_i - (len(runs)-1)/2)*0.25 for v in g['idx']]
        
        color = PALETTE[r_i % len(PALETTE)]
        
        ax.plot(x, g['score_fake_prob'], 
               marker=markers[r_i % len(markers)],
               linewidth=3.5, markersize=9,
               alpha=0.95, label=run, color=color, zorder=3)
        
        if annotate_decision:
            for xi, y in zip(x, g['score_fake_prob']):
                lab = 'F' if y >= 0.5 else 'R'
                ax.text(xi, y + 0.035, lab, ha='center', va='bottom', 
                       fontsize=11, fontweight='bold', 
                       color='red' if y >= 0.5 else '#1f77b4',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    ax.axhline(0.5, color='red', linestyle='--', linewidth=3, label='Próg 0.5')
    ax.set_xticks([idx_pos[i] for i in idxs])
    ax.set_xticklabels(range(len(idxs)), fontsize=11)
    ax.set_title(f"{source}: P(Fake) vs Audio Index", 
                fontsize=16, fontweight='bold')
    ax.set_xlabel("Audio Index (0,1,2,...)")
    ax.set_ylabel("P(Fake)")
    ax.set_ylim(-0.08, 1.08)
    ax.legend(title='Modification', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    legend_text = "\n".join([f"{i}: {label[:28]}" for i, label in enumerate(short_labels)])
    fig.text(0.83, 0.5, f"Audio Mapping:\n{legend_text}", fontsize=10.5, 
             va='center', ha='left', 
             bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=0.7'))
    
    if output_dir:
        out_file = output_dir / f"{source.replace(' ', '_')}_predictions.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {out_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare predictions from config.yaml')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config.yaml')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("📋 Config loaded:")
    print(f"   Files ({len(config['files'])}):")
    for i, f in enumerate(config['files'], 1):
        print(f"   {i}. {f}")
    print(f"   Models: {config.get('models', 'all')}")
    print(f"   Threshold: {config.get('threshold', 0.5)}")
    
    files_cfg = config.get('files', {})
    if not files_cfg:
        print("❌ No files specified in config!")
        return

    df_all, runs_labels = load_runs_with_mod(files_cfg, threshold=config.get('threshold', 0.5))
    
    output_cfg = config.get('output', {})
    output_dir = Path(output_cfg.get('result_path', 'results/Predictions/Runs_comparison'))
    output_dir = output_dir / runs_labels
    output_dir.mkdir(parents=True, exist_ok=True)
    models_to_plot = config.get('models', sorted(df_all['source'].unique()))
    runs = sorted(df_all['run'].unique(), key=lambda s: s.lower())

    print(f' Runs: {runs}')
    
    for source in models_to_plot:
        if source not in df_all['source'].values:
            print(f"⚠️ Skipping {source} (no data)")
            continue
        
        df_sub = df_all[df_all['source'] == source]
        
        idxs = sorted(df_sub['idx'].unique(), key=lambda s: try_num(s))

        idx_pos = {idx: i for i, idx in enumerate(idxs)}
        short_labels = [str(i)[:25] + '...' if len(str(i)) > 25 else str(i) for i in idxs]
        
        print(f"\n📊 Plotting {source}...")
        plot_with_right_box(df_all, source, idxs, idx_pos, PALETTE, runs, 
                           short_labels, annotate_decision=config.get('annotate', False),
                           output_dir=output_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
