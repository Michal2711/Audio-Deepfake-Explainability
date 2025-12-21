import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report, confusion_matrix

def draw_spectro(spectro, title, sr):
    fig, ax = plt.subplots()

    S_dB = librosa.amplitude_to_db(spectro, ref=np.min)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)

def run_inference(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    test_preds, test_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['spectrogram'].to(device))
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs.squeeze(1)
            preds = torch.sigmoid(outputs) > 0.5
            test_preds.append(preds.cpu())
            test_labels.append(batch["label"].cpu())

    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    return test_preds, test_labels

def plot_classification_report(test_labels, test_preds):
    print(classification_report(test_labels, test_preds, target_names=["Real", "Fake"]))
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.ylabel('Real labels')
    plt.xlabel('Predicted labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(test_labels, test_preds):
    auc = roc_auc_score(test_labels, test_preds)
    print(f"AUC: {auc:.4f}")
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()