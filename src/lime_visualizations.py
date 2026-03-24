# src/lime_visualizations.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import re

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sns
import librosa.display
import json

from feature_extraction import compute_rms_envelope, compute_rhythm_stats

def _normalize_model_name(folder_lower: str):
    """Normalize model name from folder name"""
    mappings = [
        (r'\bmusicgen\b', 'MusicGen'),
        (r'\bsuno_pro\b', 'SunoPro'),
        (r'\bsuno\b', 'Suno'),
        (r'\budio\b', 'Udio'),
        (r'\byue\b', 'YuE'),
        (r'\breal\b', 'Real'),
        (r'\belevenlabs\b', 'ElevenLabs')
    ]
    for pat, name in mappings:
        if re.search(pat, folder_lower):
            return name
    return None


def _infer_data_type(folder_lower: str):
    """Infer if data is real or generated"""
    if 'real' in folder_lower or re.search(r'\breal\b', folder_lower):
        return 'real'
    return 'generated'

def visualize_explanations(explanations, output_dir="explanations_visualizations"):
    """
    Create basic visualizations for LIME explanations.
    
    :param explanations: Dictionary with explanations
    :param output_dir: Output directory for visualizations
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    
    for folder_name, samples in explanations.items():
        data_type = "real" if "real" in folder_name.lower() else "generated"
        
        for sample_id, sample_info in samples.items():
            if sample_info.get("type") == "full_track":
                explanations_data = sample_info.get("explanations", {})
                comp_inf = explanations_data.get("component_influences")
                predicted_class = explanations_data.get("predicted_class")
                model_pred = explanations_data.get("model_prediction")
                if not comp_inf:
                    print(f"⏭️ Skipped (no component_influences) for {sample_id}")
                    continue
                for component, influence in comp_inf.items():
                    results.append({
                        'component': component,
                        'influence': influence,
                        'data_type': data_type,
                        'folder': folder_name,
                        'predicted_class': predicted_class,
                        'probability': model_pred
                    })
            elif sample_info.get("type") == "segment":
                segments = sample_info.get("segments", {})
                for seg_id, seg_data in segments.items():
                    explanations_data = seg_data.get("explanations", {})
                    comp_inf = explanations_data.get("component_influences")
                    predicted_class = explanations_data.get("predicted_class")
                    model_pred = explanations_data.get("model_prediction")
                    if not comp_inf:
                        print(f"⏭️ Skipped (no component_influences) for segment {seg_id}")
                        continue
                    for component, influence in comp_inf.items():
                        results.append({
                            'component': component,
                            'influence': influence,
                            'data_type': data_type,
                            'folder': folder_name,
                            'predicted_class': predicted_class,
                            'probability': model_pred
                        })
            else:
                print(f"⏭️ Skipped unknown type for {sample_id}")

    df = pd.DataFrame(results)
    
    if df.empty:
        print("No data to visualize")
        return
    
    base_component_order = ['vocals0', 'piano0', 'drums0', 'bass0', 'other0']
    components_present = df['component'].unique().tolist()
    component_order = [c for c in base_component_order if c in components_present] + \
                  [c for c in components_present if c not in base_component_order]
    
    # 1. Average Component Influence
    plt.figure(figsize=(12, 6))
    grouped = df.groupby(['component', 'data_type'])['influence'].mean().unstack()
    for col in ['real', 'generated']:
        if col not in grouped.columns:
            grouped[col] = np.nan
    grouped = grouped.reindex(component_order)
    
    ax = grouped.plot(kind='bar', color={'real': 'blue', 'generated': 'red'})
    plt.title('Mean Component Influence on Model Decisions')
    plt.ylabel('Mean Influence')
    plt.xlabel('Audio Component')
    plt.xticks(rotation=0)
    plt.legend(title='Data Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_component_influence.png", dpi=300)
    plt.close()
    
    # 2. Distribution of Influences
    plt.figure(figsize=(14, 8))
    for i, component in enumerate(component_order):
        plt.subplot(2, 3, i+1)
        
        comp_data = df[df['component'] == component]
        real_data = comp_data[comp_data['data_type'] == 'real']['influence']
        gen_data = comp_data[comp_data['data_type'] == 'generated']['influence']
        
        data_to_plot = []
        labels = []
        if len(real_data) > 0:
            data_to_plot.append(real_data)
            labels.append('Real')
        if len(gen_data) > 0:
            data_to_plot.append(gen_data)
            labels.append('Generated')

        if data_to_plot:
            plt.boxplot(
                data_to_plot,
                labels=labels,
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='darkblue'),
                medianprops=dict(color='red')
            )

        plt.title(f'Distribution: {component}')
        plt.ylabel('Influence Value')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle('Distribution of Component Influences by Data Type', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/influence_distributions.png", dpi=300)
    plt.close()
    
    # 3. Influence vs Probability
    plt.figure(figsize=(14, 10))
    for i, component in enumerate(component_order):
        plt.subplot(2, 3, i+1)
        
        comp_data = df[df['component'] == component]
        
        real_data = comp_data[comp_data['data_type'] == 'real']
        if not real_data.empty:
            plt.scatter(real_data['probability'], real_data['influence'],
                        alpha=0.6, color='blue', label='Real')
        
        gen_data = comp_data[comp_data['data_type'] == 'generated']
        if not gen_data.empty:
            plt.scatter(gen_data['probability'], gen_data['influence'],
                        alpha=0.6, color='red', label='Generated')

        plt.title(f'{component}')
        plt.xlabel('Probability (fake)')
        plt.ylabel('Component Influence')
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(alpha=0.3)

    plt.suptitle('Component Influence vs Classification Probability', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/influence_vs_probability.png", dpi=300)
    plt.close()

    print(f"✅ Generated visualizations in: {output_dir}")

def visualize_explanations_by_model(explanations, output_dir="explanations_visualizations"):
    """
    Create detailed visualizations per model.
    
    :param explanations: Dictionary with explanations
    :param output_dir: Output directory for visualizations
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for folder_name, samples in explanations.items():
        folder_lower = folder_name.lower()
        model_name = _normalize_model_name(folder_lower)
        data_type = _infer_data_type(folder_lower)

        if not model_name:
            continue

        for sample_id, sample_info in samples.items():
            if sample_info.get("type") == "full_track":
                explanations_data = sample_info.get("explanations", {})
                comp_inf = explanations_data.get("component_influences")
                predicted_class = explanations_data.get("predicted_class")
                model_pred = explanations_data.get("model_prediction")
                if not comp_inf:
                    print(f"⏭️ Skipped (no component_influences) for {sample_id}")
                    continue
                for component, influence in comp_inf.items():
                    results.append({
                        'model': model_name,
                        'component': component,
                        'influence': influence,
                        'data_type': data_type,
                        'predicted_class': predicted_class,
                        'probability': model_pred
                    })
            elif sample_info.get("type") == "segment":
                segments = sample_info.get("segments", {})
                for seg_id, seg_data in segments.items():
                    explanations_data = seg_data.get("explanations", {})
                    comp_inf = explanations_data.get("component_influences")
                    predicted_class = explanations_data.get("predicted_class")
                    model_pred = explanations_data.get("model_prediction")
                    if not comp_inf:
                        print(f"⏭️ Skipped (no component_influences) for segment {seg_id}")
                        continue
                    for component, influence in comp_inf.items():
                        results.append({
                            'model': model_name,
                            'component': component,
                            'influence': influence,
                            'data_type': data_type,
                            'predicted_class': predicted_class,
                            'probability': model_pred
                        })
            else:
                print(f"⏭️ Skipped unknown type for {sample_id}")

    df = pd.DataFrame(results)
    if df.empty:
        print("Empty data for visualization")
        return

    component_order = ['vocals0', 'piano0', 'drums0', 'bass0', 'other0']
    model_order = ['MusicGen', 'Suno', 'SunoPro', 'Udio', 'YuE', 'Real', 'ElevenLabs']

    # 1. Per-model visualizations
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        if len(model_df) < 3:
            continue

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.barplot(
            data=model_df,
            x='component', y='influence', hue='data_type',
            order=[c for c in component_order if c in model_df['component'].unique()],
            palette={'real': 'blue', 'generated': 'red'},
            errorbar='sd'
        )
        plt.title(f'Mean Component Influence ({model})')
        plt.ylabel('Mean Influence')
        plt.xlabel('Component')
        plt.xticks(rotation=45)
        plt.legend(title='Data Type')

        plt.subplot(1, 2, 2)
        sns.boxplot(
            data=model_df, x='predicted_class', y='probability',
            hue='data_type', palette={'real': 'blue', 'generated': 'red'}
        )
        plt.title(f'Probability Distribution ({model})')
        plt.ylabel('Fake Probability')
        plt.xlabel('Predicted Class')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_comparison.png", dpi=300)
        plt.close()

    # 2. Model Comparison
    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=df, x='model', y='influence', hue='component',
        order=[m for m in model_order if m in df['model'].unique()],
        hue_order=[c for c in component_order if c in df['component'].unique()],
        errorbar='sd', palette='viridis'
    )
    plt.title('Component Influence Comparison Across Models')
    plt.ylabel('Mean Influence')
    plt.xlabel('Model')
    plt.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_models_comparison.png", dpi=300)
    plt.close()

    # 3. Heatmap
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot_table(
        index=['model', 'data_type'],
        columns='component', values='influence', aggfunc='mean'
    )
    pivot_df = pivot_df.reindex(columns=[c for c in component_order if c in pivot_df.columns])
    sns.heatmap(
        pivot_df, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, cbar_kws={'label': 'Mean Influence'}
    )
    plt.title('Mean Component Influence by Model and Data Type')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/influence_heatmap.png", dpi=300)
    plt.close()

    # 4. Facet Grid
    g = sns.FacetGrid(
        df, col='model', hue='component',
        col_order=[m for m in model_order if m in df['model'].unique()],
        col_wrap=3, height=4, aspect=1.2
    )
    g.map(sns.scatterplot, 'probability', 'influence', alpha=0.7)
    g.add_legend(title='Component')
    g.set_axis_labels("Fake Probability", "Component Influence")
    g.fig.suptitle('Component Influence vs Classification Probability', y=1.05)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/probability_vs_influence.png", dpi=300)
    plt.close()

    print(f"✅ Generated visualizations in: {output_dir}")

def plot_waveforms_overlay_with_influences(
    original_audio,
    components,
    component_names,
    influences,
    sr,
    output_path,
    prefix="",
    figsize=(13, 6)
):
    plt.figure(figsize=figsize)
    duration = len(original_audio) / sr
    times = np.linspace(0, duration, len(original_audio))

    # Gray original waveform
    plt.plot(times, original_audio, color="grey", linewidth=1.1, alpha=0.55, label="Original")

    color_map = {
        "vocals0": "red",
        "drums0": "blue",
        "bass0": "green",
        "piano0": "orange",
        "other0": "purple"
    }

    colors = list(color_map.values())
    next_color_idx = 0

    for comp, audio in zip(component_names, components):
        if len(audio) < len(times):
            audio_to_plot = np.pad(audio, (0, len(times) - len(audio)), mode='constant')
        else:
            audio_to_plot = audio[:len(times)]
        influence = influences.get(comp, None)
        color = color_map.get(comp, colors[next_color_idx % len(colors)])
        next_color_idx += 1
        infl_str = f"{influence:.3f}" if influence is not None else "N/A"
        label = f"{comp} (influence: {infl_str})"
        plt.plot(times, audio_to_plot, color=color, alpha=0.8, label=label, linewidth=1.15)

    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Waveforms Overlayed with Influences{f' ({prefix})' if prefix else ''}")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    outfile = Path(output_path) / f"{prefix}_waveforms_overlay_influences.png"
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

def plot_stacked_rms_area_components(
    components,
    component_names,
    influences,
    sr,
    output_path,
    prefix="",
    frame_length=2048,
    hop_length=2048
):
    color_map = {
        "vocals0": "#E63946",
        "drums0": "#457B9D",
        "bass0": "#1D3557",
        "piano0": "#F4A261",
        "other0": "#A8DADC"
    }
    plt.figure(figsize=(15,6))

    rms_all = []
    times_all = []
    for comp, audio in zip(component_names, components):
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        rms_all.append(rms)
        times_all.append(t)

    min_len = min(len(r) for r in rms_all)
    rms_all = np.stack([r[:min_len] for r in rms_all], axis=0)
    stacked = np.cumsum(rms_all, axis=0)
    times = times_all[0][:min_len]

    base = np.zeros_like(times)
    for i, comp in enumerate(component_names):
        color = color_map.get(comp, None)
        infl = influences.get(comp, 0)
        label = f"{comp} (influence: {infl:.3f})"
        plt.fill_between(times, base, stacked[i], color=color, alpha=0.72, label=label)
        base = stacked[i]

    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative RMS amplitude")
    plt.title(f"Stacked RMS Energy Per Component{f' ({prefix})' if prefix else ''}")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    outfile = Path(output_path) / f"{prefix}_stacked_rms_area_components.png"
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()

def plot_bar_influences_per_sample(
    component_influences,
    output_path,
    prefix="",
    component_order=None,
    predicted_class="Unknown",
    model_name="",
    model_pred=0.0
):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if component_order is None:
        component_order = ['vocals0', 'drums0', 'bass0', 'other0']

    influences_dict = {comp: component_influences.get(comp, 0.0) for comp in component_order}
    influences_list = [influences_dict[comp] for comp in component_order]

    plt.figure(figsize=(10, 6))
    colors = ["red" if x < 0 else "green" for x in influences_list]
    plt.bar(component_order, influences_list, color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel("Audio Components")
    plt.ylabel("Influence on Model Decision")
    title_prefix = f"{model_name}/" if model_name else ""
    plt.title(
        f"LIME Influences (Bar): {title_prefix}{prefix}\n"
        f"Class: {predicted_class}, P(fake): {model_pred:.3f}"
    )
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    outfile = Path(output_path) / f"{prefix}_bar_influences.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved BAR: {outfile}")

def plot_radar_influences_per_sample(
    component_influences,
    output_path,
    prefix="",
    component_order=None,
    predicted_class="Unknown",
    model_name=""
):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if component_order is None:
        component_order = ['vocals0', 'drums0', 'bass0', 'other0']

    influences_dict = {comp: component_influences.get(comp, 0.0) for comp in component_order}
    influences_list = [influences_dict[comp] for comp in component_order]

    signs = ['+' if infl >= 0 else '−' for infl in influences_list]
    colors = ['green' if s == '+' else 'red' for s in signs]
    abs_values = [abs(infl) for infl in influences_list]

    angles = np.linspace(0, 2 * np.pi, len(component_order), endpoint=False).tolist()
    angles += angles[:1]
    abs_values += abs_values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0], angle=30, fontsize=10, alpha=0.7)
    ax.set_ylim(0, 1.0)
    ax.fill(angles, abs_values, color='skyblue', alpha=0.35)
    ax.plot(angles, abs_values, color='navy', linewidth=4, marker='o', markersize=14)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(component_order, fontsize=14, weight='bold')

    
    from matplotlib.patheffects import withStroke
    
    for i, (angle, infl, sign) in enumerate(zip(angles[:-1], influences_list, signs)):
        label_text = f'{sign}{abs(infl):.3f}'
        
        
        r_distance = 1.45 + (abs(infl) * 0.1)
        
        ax.annotate(
            label_text, 
            xy=(angle, r_distance),
            ha='center', va='center',
            fontsize=8,
            fontweight='bold', 
            color=colors[i],
            path_effects=[withStroke(foreground="white", linewidth=3)],
            bbox=dict(boxstyle="round,pad=0.15", facecolor='white', alpha=0.95),
            clip_on=False
        )

    # Legenda na dole
    legend_elements = []
    for i, comp in enumerate(component_order):
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                       markersize=12, label=f'{comp}: {signs[i]}{influences_list[i]:.3f}')
        )
    
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=11, frameon=True, fancybox=True)

    title_prefix = f"{model_name} / " if model_name else ""
    ax.set_title(
        f"LIME Component Influences (Radar Plot)\n"
        f"{title_prefix}{prefix} | Class: {predicted_class}",
        fontsize=16, pad=30, weight='bold', color='darkblue'
    )
    
    plt.tight_layout()
    outfile = Path(output_path) / f"{prefix}_radar_influences.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved RADAR: {outfile}")

def visualize_per_sample_explanations(
    explanations, 
    features_outputdir_full="features_outputdir_full",
    max_samples_per_model=10
):
    features_base = Path(features_outputdir_full)
    COMPONENT_ORDER = ['vocals0', 'drums0', 'bass0', 'other0']
    
    print(f"🔍 Generating per-sample visualizations (BAR + RADAR) in: {features_base}")

    sample_count = 0
    for model_folder, samples in explanations.items():
        model_name = _normalize_model_name(model_folder.lower())
        if not model_name:
            continue
        
        model_dir = features_base / model_folder
        
        for sample_id, sample_info in list(samples.items())[:max_samples_per_model]:
            if sample_info.get("type") != "full_track":
                continue
            
            explanations_data = sample_info.get("explanations", {})
            comp_inf = explanations_data.get("component_influences", {})
            predicted_class = explanations_data.get("predicted_class", "Unknown")
            model_pred = explanations_data.get("model_prediction", 0.0)
            
            if not comp_inf or len(comp_inf) == 0:
                print(f"⏭️ Skipped (no influences): {model_folder}/{sample_id}")
                continue
            
            safename = sample_id 
            sample_output_dir = model_dir / safename
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. BAR PLOT
            plot_bar_influences_per_sample(
                component_influences=comp_inf,
                output_path=sample_output_dir,
                prefix=safename,
                component_order=COMPONENT_ORDER,
                predicted_class=predicted_class,
                model_name=model_name,
                model_pred=model_pred,
            )
            
            # 2. RADAR PLOT
            plot_radar_influences_per_sample(
                component_influences=comp_inf,
                output_path=sample_output_dir,
                prefix=safename,
                component_order=COMPONENT_ORDER,
                predicted_class=predicted_class,
                model_name=model_name,
            )
            
            sample_count += 1
            print(f"✅ Saved BAR+RADAR: {model_folder}/{safename}/")
    
    print(f"✅ Offline visualizations for {sample_count} samples in: {features_base}")
