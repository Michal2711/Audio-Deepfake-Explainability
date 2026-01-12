# Normalne uruchomienie (z auto-resume)
python scripts/run_FBP_experiment.py

# Wymuś przetworzenie wszystkich plików od nowa
python scripts/run_FBP_experiment.py --force-reprocess

# Jawnie wznów od checkpointu
python scripts/run_FBP_experiment.py --resume

# Wyłącz checkpointing (nie zalecane)
python scripts/run_FBP_experiment.py --no-checkpoint

# Zobacz listę błędnych plików
python scripts/run_FBP_experiment.py --show-failed

# Użyj własnych plików konfiguracyjnych
python scripts/run_FBP_experiment.py --base configs/my_base.yaml --exp configs/my_exp.yaml

# Jeśli eksperyment zostanie przerwany (Ctrl+C), po prostu uruchom ponownie:
python scripts/run_FBP_experiment.py
# Automatycznie wznowi od miejsca przerwania

--------------------------------------------------------------
-- LIME

# Uruchom z domyślną konfiguracją
python scripts/run_LIME_experiment.py

# Użyj własnej konfiguracji
python scripts/run_LIME_experiment.py --config configs/my_lime_config.yaml

# Wyłącz checkpointing
python scripts/run_LIME_experiment.py --no-checkpoint

# Wznów od checkpointu
python scripts/run_LIME_experiment.py --resume

--------------------------------------------------------------
-- Occlusion

# Użyj własnej konfiguracji
python .\scripts\run_spectrogram_experiment.py --config .\configs\Spec_occlusion_configs\spectrogram_explainability.yaml

# Oblicz cechy fizyczne wyników Occlusion
python .\scripts\run_occlusion_patch_features.py --config .\configs\Spec_occlusion_configs\occlusion_patch_features.yaml