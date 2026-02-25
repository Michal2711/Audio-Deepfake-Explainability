# Audio Explainability Experiments for Fake Song Detection

This repository contains the experimental codebase developed for a master’s thesis on explainable artificial intelligence (XAI) for audio-based fake song detection. It implements three complementary explanation methods for a spectrogram-based classifier and provides pipelines for computing physically interpretable audio features and visualizing their importance.

The experiments operate on the same dataset of real and synthetic music tracks and use a shared spectrogram-based model, either via a local checkpoint (`awsaf49/sonics-spectttra-alpha-120s`) or a remote API served from `awsaf49/sonics-fake-song-detection` (model type `SpecTTTra-α`).

---

## 1. Implemented Explainability Methods

### 1.1 AudioLIME (Source-Separation-Based Explanations)

**Goal:** Explain model predictions by perturbing and recombining separated audio stems (e.g., vocals, drums, bass) and measuring their influence on the classifier output.

- Implemented in: `src/lime_explainer` with running script store int `scripts/experiments/run_LIME_experiment.py`
- Main configuration: `configs/lime_experiment.yaml`
- Uses LIME on top of track-level predictions, with optional visualization of global and per-model explanations.
- Designed to process a configurable subset of tracks (`max_samples_explain`) and multiple model categories (e.g., UDIO, SUNO, REAL, ElevenLabs).

### 1.2 FBP – Frequency Band Perturbation

**Goal:** Explain predictions by selectively attenuating or removing predefined frequency bands in the spectrogram and observing the impact on model confidence.

- Implemented in: `src/dsp_band_ops` with runnign script in `scripts/experiments/run_FBP_experiment.py`
- Main configuration: `configs/fbp_experiment.yaml`
- Operates on a dataset grouped by generator model (UDIO, SUNO, SUNO_PRO, ElevenLabs, REAL) with a limit on samples per model.
- Uses a configurable bank of frequency bands (e.g., `default`, `detailed_voice`, `high_resolution`) and an attenuation factor (e.g., 0.25 = 75% energy reduction).
- Supports both mixture-level and (optionally) source-separated analysis, as well as STFT or mel spectrograms.

### 1.3 Occlusion – Spectrogram Importance Maps

**Goal:** Produce 2D saliency maps on the time–frequency plane by occluding patches of the spectrogram and measuring the change in model output.

- Implemented in: `scr/spectrogram_explainability` with running script in `scripts/experiments/run_spectrogram_experiment.py`
- Main configuration: `configs/spectrogram_explainability.yaml`
- Uses a sliding occlusion window defined in spectrogram frames and frequency percentage (e.g., 1024 time frames and 5% of the frequency axis, with configurable stride).
- Can optionally store the most influential occluded windows as reconstructed audio segments, allowing auditory inspection of critical regions.
- Produces importance maps that highlight the most influential regions for a given prediction, with configurable highlight percentage for visualization.

## 2. Repository Structure

The main project structure of scripts is organized as follows:

```text
scripts/
├── experiments/
│   ├── run_FBP_experiment.py
│   ├── run_LIME_experiment.py
│   └── run_spectrogram_experiment.py
├── feature_extraction/
│   ├── run_FBP_patch_features.py
│   ├── run_occlusion_patch_features.py
│   └── run_features_extraction.py
└── feature_visualizations/
    ├── run_FBP_features_vis.py
    ├── run_Occlusion_features_vis.py
    └── run_sonics_predictions.py
```


## 3. Configuration Files

All experiments are driven by YAML configuration files located in `configs/`.

### 3.1. AudioLIME Configuration (`configs/AudioLIME_configs/lime_experiment.yaml`)

- **Dataset:** Path to audio (`dataset_path`), typically a directory with read and generated tracks.
- **Model:** Choice between local or remote model, device selection (`cuda`/`cpu`), retry policy and model time window (e.g. 120s).
- **LIME parameters:** Number of samples for LIME (`num_samples_lime`), maximum number of tracks to explain 
- **Output and visualization:** Configure the output directory, experiment name and whether to generate overall or per-model summaries of the explanations.

### 3.2. FBP Configuration (`configs/FBP_configs/fbp_experiment.yaml`)

- **Dataset:** Defines the base path to the dataset, the list of generator models, and the maximum number of samples per model.

- **Spectrogram:** Describes the spectrogram type (e.g., STFT or mel), sampling rate, FFT parameters, and analysis duration.

- **Frequency bands:** Specifies band presets and the attenuation factor applied to each band when computing explanations.

- **Explainability options:** Provide optional support for source separation and loudness normalization.

- **Output and checkpointing:** Define the experiment name, result path, and explicit checkpoint settings to enable resuming long runs.

### 3.3 Occlusion Configuration (`configs/Spec_occlusion_configs/spectrogram_explainability.yaml`)

- **Dataset and model:** Follow the same pattern as FBP, ensuring that the same classifier and dataset organization are used.

- **Explainability method:** Selects occlusion as the explanation technique and defines basic thresholds for processing predictions.

- **Occlusion parameters:** Configure the patch size and stride in both time and frequency, as well as how many of the most important windows are stored as audio examples.

- **Visualization:** Controls how much of the most important area of the spectrogram is highlighted in resulting saliency maps.

- **Output and checkpointing:** Specify the output directory, experiment name, and checkpoint settings.

## 4. Running the Experiments
Each experiment supports automatic checkpointing and resumption. If execution is interrupted (for example, using Ctrl+C), rerunning the corresponding command continues processing from the last saved checkpoint.

### 4.1 AudioLIME Experiment

Run with the default configuration:

```bash
python scripts/experiments/run_LIME_experiment.py
```

Use a custom configuration file:

```bash
python scripts/experiments/run_LIME_experiment.py \
--config configs/my_lime_experiment.yaml
```
Disable checkpointing:

```bash
python scripts/experiments/run_LIME_experiment.py --no-checkpoint
```

Resume from a checkpoint:

```bash
python scripts/experiments/run_LIME_experiment.py --resume
```

### 4.2 FBP Experiment
Normal run with automatic resume:

```bash
python scripts/experiments/run_FBP_experiment.py
```

Explicitly resume from a checkpoint:

```bash
python scripts/experiments/run_FBP_experiment.py --resume
```

Disable checkpointing (not recommended for long experiments):

```bash
python scripts/experiments/run_FBP_experiment.py --no-checkpoint
```

Use custom configuration files:

```bash
python scripts/experiments/run_FBP_experiment.py \
  --config configs/FBP_configs\fbp_experiment.yaml \
```

Just creating global_visualizations:
```bash
python scripts/experiments/run_FBP_experiment.py \
  --config configs/FBP_configs\fbp_experiment.yaml \
  --visualize-only .\results\FBP\[Dataset]\[experiment_name]\fbp_results.json
  --bands-root .\results\FBP\[Dataset]\[experiment_name]\bands\
```


### 4.3 Occlusion Experiment (Spectrogram Explainability)
Run the occlusion-based spectrogram explainability experiment:

```bash
python scripts/experiments/run_spectrogram_experiment.py \
  --config configs/spectrogram_explainability.yaml
```

## 5. Feature Extraction Pipelines
For each experiment, the repository provides scripts that compute physically interpretable audio features based on the audio generated or modified during the explainability analyses.

### 5.1 AudioLIME Feature Extraction
For AudioLIME, feature extraction is performed on both the original track and the separated stems obtained during source separation:

```bash
python scripts/feature_extraction/run_features_extraction.py \
  --config configs/Features_extraction/features_configs.yaml
```

This script can be configured to compute a set of acoustic descriptors on full tracks and individual stems, enabling comparison of the contribution of different musical components (e.g., vocals vs. accompaniment).

### 5.2 FBP Feature Extraction
After running the FBP experiment and saving band-perturbed audio files, patch-level physical features can be extracted using:

```bash
python scripts/feature_extraction/run_FBP_patch_features.py \
  --config configs/FBP_configs/fbp_bands_features.yaml
```
This converts band attenuation experiments into structured feature vectors that can be analyzed statistically, for example to quantify which frequency regions are most strongly associated with the model’s fake/real decisions.

### 5.3 Occlusion Feature Extraction
For the Occlusion method, physical features are computed for the most important occluded spectrogram regions and corresponding audio segments:

```bash
python scripts/feature_extraction/run_occlusion_patch_features.py \
  --config configs/Spec_occlusion_configs/occlusion_patch_features.yaml
```
The resulting feature tables summarize time–frequency regions that strongly influence the classifier, which supports systematic comparison across generator models and classes.

## 6. Visualization of Physical Features
The repository includes dedicated scripts for visualizing the extracted features and relating them to model decisions.

### 6.1 FBP Feature Visualization
```bash
python scripts/feature_visualizations/run_FBP_features_vis.py --config configs/FBP_configs/fbp_features_vis.yaml
```
This script produces plots such as band-wise importance distributions, aggregated statistics per generator model, and comparisons between real and synthetic content.

### 6.2 Occlusion Feature Visualization
```bash
python scripts/feature_visualizations/run_Occlusion_features_vis.py --config configs/Spec_occlusion_configs/occlusion_features_vis.yaml
```
Here, the focus is on visualizing time–frequency importance patterns and distributions of features computed on the most relevant occluded regions.

### 6.3 Auxiliary Predictions
```bash
python scripts/run_sonics_predictions.py
```
This auxiliary script can be used to generate baseline predictions from the spectrogram classifier on a chosen dataset, providing a reference point for the explainability experiments.

