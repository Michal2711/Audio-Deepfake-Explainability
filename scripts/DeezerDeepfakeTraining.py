import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from tqdm import tqdm

class DeezerDeepfakeTrainer:
    def __init__(self, model, device, train_loader, val_loader=None, use_spectrogram=False, save_dir="./models", patience=5, learning_rate=1e-3, neptune_run=None):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_spectrogram = use_spectrogram
        self.save_dir = save_dir
        self.patience = patience
        self.neptune_run = neptune_run

        os.makedirs(self.save_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        if self.neptune_run:
            self.neptune_run["parameters"] = {
                "optimizer": "Adam",
                "initial_lr": learning_rate,
                "scheduler": "ReduceLROnPlateau",
                "scheduler_patience": 2,
                "early_stopping_patience": patience,
                "use_spectrogram": use_spectrogram,
            }

    def process_batch(self, batch_audio, batch_labels):
        batch_audio, batch_labels = batch_audio.to(self.device), batch_labels.to(self.device)

        outputs = self.model(batch_audio)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        outputs = outputs.squeeze(1)
        loss = self.criterion(outputs, batch_labels)

        preds = torch.sigmoid(outputs) > 0.5

        return loss, preds, batch_labels

    def train_one_epoch(self):
        self.model.train()
        all_preds = []
        all_labels = []
        total_loss = 0

        progress = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress:
            self.optimizer.zero_grad()
            loss, preds, labels = self.process_batch(batch['spectrogram'], batch['label'] if self.use_spectrogram else (batch['audio'], batch['label']))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            progress.set_postfix(loss=loss.item())
            if self.neptune_run:        
                self.neptune_run["train/loss"].log(loss.item())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        try:
            auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        except ValueError:
            auc = 0.5

        return total_loss / len(self.train_loader), acc, auc

    def validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        progress = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for batch in progress:
                loss, preds, labels = self.process_batch(batch['spectrogram'], batch['label'] if self.use_spectrogram else (batch['audio'], batch['label']))
                total_loss += loss.item()
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())
                progress.set_postfix({"val_loss": loss.item()})

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
        try:
            auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        except ValueError:
            auc = 0.5

        return total_loss / len(self.val_loader), acc, auc

    def fit(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, train_auc = self.train_one_epoch()
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")

            if self.neptune_run:
                self.neptune_run["train/loss"].log(train_loss)
                self.neptune_run["train/acc"].log(train_acc)
                self.neptune_run["train/auc"].log(train_auc)

            if self.val_loader:
                val_loss, val_acc, val_auc = self.validate()
                print(f"          - Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

                if self.neptune_run:
                    self.neptune_run["val/loss"].log(val_loss)
                    self.neptune_run["val/acc"].log(val_acc)
                    self.neptune_run["val/auc"].log(val_auc)
                    self.neptune_run["lr"].log(self.optimizer.param_groups[0]['lr'])

                self.scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    best_model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), best_model_path)
                    print("          - Saved new best model!")

                    if self.neptune_run:
                        self.neptune_run["best_model"].upload(best_model_path)
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                    break
