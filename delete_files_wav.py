import os
import shutil
from pathlib import Path

def delete_files_by_extension(root_dir, extensions):
    root = Path(root_dir)
    if not root.is_dir():
        print(f"Podana ścieżka nie jest katalogiem: {root_dir}")
        return

    count = 0
    for ext in extensions:
        pattern = f"**/*{ext}"
        for file_path in root.glob(pattern):
            if file_path.is_file():
                print(f"Usuwanie: {file_path}")
                file_path.unlink()
                count += 1

    print(f"Usunięto {count} plików.")

def delete_folder(folder_path):
    """Usuwa cały katalog (zawartość + podkatalogi), jeśli istnieje."""
    folder = Path(folder_path)
    if folder.exists() and folder.is_dir():
        print(f"Usuwanie folderu: {folder}")
        shutil.rmtree(folder)  # UWAGA: usuwa nieodwracalnie cały katalog![web:17][web:22]
    else:
        print(f"Folder nie istnieje lub nie jest katalogiem: {folder}")


def main():
    print("Usuwanie plików .wav i .png z katalogu 'mixture'...")
    dataset = "FakeRealMusicOriginal/"
    lufs = ''
    perturbation = ''
    experiment_name = "FBP_default_preset_rel_stft_att_0.25_mixture"

    folder_path = f"results/FBP/{dataset}{lufs}{perturbation}{experiment_name}/bands"
    
    for folder in os.listdir(folder_path):
        print(f"Przetwarzanie folderu: {folder}")
        model_path = os.path.join(folder_path, folder)
        for audio_file in os.listdir(model_path):
            print(f"Przetwarzanie folderu pliku: {audio_file}")
            audio_file_path = os.path.join(model_path, audio_file)
            print(audio_file_path)
            comp_path = Path(audio_file_path) / "mixture"
            batch_vis_path = Path(comp_path) / "batches_vis"
            freq_batches_path = Path(comp_path) / "freq_batches"

            delete_folder(batch_vis_path)
            delete_folder(freq_batches_path)

            # for file in os.listdir(comp_path):
            #     file_path = os.path.join(comp_path, file)
            #     print(f"Przetwarzanie batches: {file_path}")
            #     if os.path.isdir(file_path):
            #         print(f"Przetwarzanie folderu: {file_path}")
            #         delete_files_by_extension(file_path, [".wav", ".png"])

if __name__ == "__main__":
    main()