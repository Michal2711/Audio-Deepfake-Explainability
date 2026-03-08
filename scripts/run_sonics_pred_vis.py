# scripts/run_sonics_pred_vis.py
import json
import argparse
from pathlib import Path
import yaml

import numpy as np
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
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

def load_yaml(path: Path):
    with open(path, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_predictions(json_path):
    """Wczytuje predictions.json - struktura jak w Twoim pliku"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = []
    for model, tracks in data.items():
        for track_id, info in tracks.items():
            row = {
                'model': model,
                'track_id': track_id,
                'track_stem': info.get('track_stem', track_id),
                'prediction': info.get('prediction', np.nan),
                'predicted_class': info.get('predicted_class', 'Unknown'),
                'true_class': info.get('track_source', 'Unknown'),
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')
    
    class_map = {'Real': 0, 'Fake': 1, 'REAL': 0, 'GENERATED': 1}
    df['true_binary'] = df['true_class'].map(class_map).fillna(-1)
    df['pred_binary'] = df['predicted_class'].map(class_map).fillna(-1)
    
    return df

def plot_model_predictions_lines(df, models, colors, config, output_dir):

    setup_professional_style()
    out_dir = output_dir / "model_predictions_clean"
    out_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, model in enumerate(models[:5]):
        ax = axes[i]
        
        df_model = df[df['model'] == model].copy()
        unique_tracks = sorted(df_model['track_stem'].unique())
        track_to_idx = {track: j for j, track in enumerate(unique_tracks)}
        
        df_model['track_idx'] = df_model['track_stem'].map(track_to_idx)
        df_model = df_model.sort_values('track_idx')
        
        x = df_model['track_idx'].values
        y = df_model['prediction'].values
        color = colors.get(model, '#7f7f7f')
        
        ax.plot(x, y, linewidth=5, color=color, alpha=0.95, zorder=3)
        
        ax.axhline(0.5, color='red', linestyle='--', linewidth=3, alpha=0.9)
        
        ax.set_xticks(range(len(unique_tracks)))
        ax.set_xticklabels(range(len(unique_tracks)), fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{model}', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Audio Track Index')
        ax.set_ylabel('P(Fake)')
        ax.grid(True, alpha=0.25)
    
    for i in range(len(models), 6):
        fig.delaxes(axes[i])
    
    plt.suptitle('SONICS Model Predictions: P(Fake) Confidence per Audio Track\n'
                '(Decision threshold 0.5)', 
                fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')
    
    plt.tight_layout()

    out_file = out_dir / f"predictions_lines.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved predictions lines: {out_file}")

def plot_confusion_matrices(df, models, colors, config, output_dir):
    setup_professional_style()
    out_dir = output_dir / "confusion_matrices"
    out_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        df_model = df[(df['model'] == model) & 
                     (df['true_binary'] != -1) & 
                     (df['pred_binary'] != -1)]
        
        cm = confusion_matrix(df_model['true_binary'], df_model['pred_binary'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        axes[i].set_title(f'{model}\nACC: {accuracy_score(df_model["true_binary"], df_model["pred_binary"]):.3f}')
    
    plt.suptitle('Confusion Matrices per Model', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_file = out_dir / f"confusion_matrices.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrices saved: {out_file}")

def plot_metrics_table(df, models, config, output_dir):
    setup_professional_style()
    out_dir = output_dir / "metrics"
    out_dir.mkdir(exist_ok=True)
    
    metrics_data = []
    for model in models:
        df_model = df[(df['model'] == model) & 
                     (df['true_binary'] != -1) & 
                     (df['pred_binary'] != -1)]
        
        if len(df_model) > 0:
            y_true, y_pred = df_model['true_binary'], df_model['pred_binary']
            metrics_data.append({
                'Model': model,
                'ACC': f'{accuracy_score(y_true, y_pred):.3f}',
                'PREC': f'{precision_score(y_true, y_pred, zero_division=0):.3f}',
                'REC': f'{recall_score(y_true, y_pred, zero_division=0):.3f}',
                'F1': f'{f1_score(y_true, y_pred, zero_division=0):.3f}',
                'N': f'{len(df_model)}'
            })
    
    if not metrics_data:
        print("⚠️ No metrics data")
        return
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig, ax = plt.subplots(figsize=(10, len(metrics_data)*0.35 + 1.5))
    ax.axis('off')
    
    table = ax.table(cellText=metrics_df.iloc[:, 1:].values,
                    colLabels=metrics_df.columns[1:],
                    rowLabels=metrics_df['Model'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 0.9])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 2.2)
    
    n_cols = len(metrics_df.columns) - 1
    for j in range(n_cols):
        hdr = table[(0, j)]
        hdr.set_facecolor('#2E86C1')
        hdr.set_text_props(weight='bold', color='white')
        hdr.set_height(0.1)
    
    for i in range(1, len(metrics_data) + 1):
        for j in range(n_cols):
            cell = table[(i, j)]
            cell.set_facecolor('#F8F9FA' if i % 2 == 0 else '#FFFFFF')
            cell.set_edgecolor('#DEE2E6')
    
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1.5)
        cell.set_edgecolor('#495057')
    
    plt.title('SONICS Model Performance Summary', 
             fontsize=18, fontweight='bold', pad=30, color='#2E86C1')
    
    fig.text(0.5, 0.02, 'ACC=Accuracy, PREC=Precision, REC=Recall, F1=F1-Score, N=Number of samples', 
            ha='center', fontsize=10, style='italic')
    
    out_file = out_dir / f"performance_metrics.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Metrics saved: {out_file}")


def plot_threshold_curves(df, models, colors, config, output_dir):    
    setup_professional_style()
    out_dir = output_dir / "threshold_curves"
    out_dir.mkdir(exist_ok=True)
    
    thresholds = np.linspace(0.0, 1.0, 101)
    all_shares = []
    
    for model in models:
        df_model = df[df['model'] == model].copy()
        if len(df_model) == 0 or df_model['prediction'].isna().all():
            continue
            
        shares = []
        for thr in thresholds:
            share_fake = (df_model['prediction'] >= thr).mean()
            shares.append({'threshold': thr, 'share_fake': share_fake, 'model': model})
        
        model_shares = pd.DataFrame(shares)
        all_shares.append(model_shares)
    
    curve_df = pd.concat(all_shares, ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    for model in models:
        df_model_curve = curve_df[curve_df['model'] == model]
        color = colors.get(model, 'gray')
        
        ax.plot(df_model_curve['threshold'], df_model_curve['share_fake'], 
               color=color, linewidth=4.5, alpha=0.95,
               label=model)
    
    ax.axvline(0.5, color='red', linestyle='--', linewidth=3, alpha=0.9, label='Threshold 0.5')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Threshold P(Fake)', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel("Share of 'Fake' Decisions", fontsize=14, fontweight='bold', labelpad=12)
    ax.set_title("Share of 'Fake' Decisions vs. Threshold per Source", 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    plt.tight_layout()
    
    out_file = out_dir / f"fake_share_vs_threshold_professional.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Threshold curves saved: {out_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize SONICS predictions')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to config.yaml')
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    
    print("📊 Loading predictions...")
    data_cfg = config.get("data", {})
    predictions_path = data_cfg.get("predictions_path")
    df = load_predictions(predictions_path)
    print(f"   Loaded {len(df)} predictions for {len(df['model'].unique())} models")
    
    output_cfg = config.get('output', {})
    output_root = output_cfg.get('result_path')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    vis_cfg = config.get('visualizations', {})
    models = vis_cfg.get('models', {})
    colors = vis_cfg.get('colors', {})
    
    print("\Generating visualizations...")
    
    plot_model_predictions_lines(df, models, colors, config, output_root)
    plot_confusion_matrices(df, models, colors, config, output_root)
    plot_metrics_table(df, models, config, output_root)
    plot_threshold_curves(df, models, colors, config, output_root)  # 🆕 z models, colors
    
    print(f"\n✅ All visualizations saved to: {output_root}")

if __name__ == "__main__":
    main()
