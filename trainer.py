import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np


class Trainer:
    """
    Ogólna klasa do trenowania modeli w PyTorch.
    Obsługuje specyfikę FaceReIDModel (maskowanie niewykrytych twarzy).
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        optimizer=None,
        criterion=None,
        device="cpu",
        save_dir="checkpoints",
    ):
        """
        Args:
            model (nn.Module): Model (z głowicą klasyfikacyjną) do trenowania.
            train_loader (DataLoader): DataLoader zbioru treningowego.
            val_loader (DataLoader, optional): DataLoader zbioru walidacyjnego.
            optimizer (torch.optim): Optymalizator (np. Adam).
            criterion (nn.Module): Funkcja straty (np. CrossEntropyLoss).
            device (str): 'cuda' lub 'cpu'.
            save_dir (str): Katalog do zapisu wag modelu.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def train_epoch(self, epoch_idx):
        """Przeprowadza jedną epokę treningową."""
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_valid_samples = 0

        # Pasek postępu
        pbar = tqdm(self.train_loader, desc=f"Train Epoka {epoch_idx}")

        for batch in pbar:
            # Obsługa specyficznego Datasetu VC-Clothes (zwraca 4 elementy)
            # batch[0]=images, batch[1]=pids, batch[2]=camids, batch[3]=clothes_ids
            images = batch[0]  # Lista PIL images (dzięki custom collate_fn) lub Tensor
            labels = batch[1].to(self.device)

            self.optimizer.zero_grad()

            # --- Forward pass ---
            output = self.model(images)

            # --- Obsługa specyfiki FaceReIDModel ---
            # Model może zwrócić krotkę (logits, valid_mask) jeśli używamy detekcji twarzy
            if isinstance(output, tuple):
                logits, valid_mask = output

                # Jeśli w całym batchu nie wykryto twarzy, pomijamy krok
                if logits is None or valid_mask.sum() == 0:
                    continue

                # Filtrujemy tylko poprawne próbki
                final_logits = logits[valid_mask]
                final_labels = labels[valid_mask]
            else:
                # Standardowy model
                final_logits = output
                final_labels = labels

            # --- Loss & Backward ---
            loss = self.criterion(final_logits, final_labels)
            loss.backward()
            self.optimizer.step()

            # --- Statystyki ---
            batch_size = final_labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = torch.max(final_logits, 1)
            correct_preds += (predicted == final_labels).sum().item()
            total_valid_samples += batch_size

            # Aktualizacja paska postępu
            current_acc = (
                correct_preds / total_valid_samples if total_valid_samples > 0 else 0
            )
            pbar.set_postfix({"loss": loss.item(), "acc": current_acc})

        epoch_loss = (
            running_loss / total_valid_samples if total_valid_samples > 0 else 0
        )
        epoch_acc = (
            correct_preds / total_valid_samples if total_valid_samples > 0 else 0
        )

        return epoch_loss, epoch_acc

    def validate(self, epoch_idx):
        """Przeprowadza walidację (opcjonalne)."""
        if not self.val_loader:
            return 0.0, 0.0

        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Valid Epoka {epoch_idx}")

        with torch.no_grad():
            for batch in pbar:
                images = batch[0]
                labels = batch[1].to(self.device)

                output = self.model(images)

                if isinstance(output, tuple):
                    logits, valid_mask = output
                    if logits is None:
                        continue
                    final_logits = logits[valid_mask]
                    final_labels = labels[valid_mask]
                else:
                    final_logits = output
                    final_labels = labels

                if len(final_labels) == 0:
                    continue

                loss = self.criterion(final_logits, final_labels)

                running_loss += loss.item() * final_labels.size(0)
                _, predicted = torch.max(final_logits, 1)
                correct_preds += (predicted == final_labels).sum().item()
                total_samples += final_labels.size(0)

        val_loss = running_loss / total_samples if total_samples > 0 else 0
        val_acc = correct_preds / total_samples if total_samples > 0 else 0

        print(f"Walidacja: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        return val_loss, val_acc

    def fit(self, num_epochs):
        """Główna pętla sterująca treningiem."""
        print(f"Rozpoczęcie treningu na urządzeniu: {self.device}")

        best_acc = 0.0
        results = []

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            results.append(train_loss, train_acc, val_loss, val_acc)

            print(
                f"Epoka {epoch}/{num_epochs} zakończona. Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

            # Zapisywanie checkpointu (co epokę)
            self.save_checkpoint(epoch, final=False)

            # Zapisywanie najlepszego modelu (jeśli mamy walidację)
            if self.val_loader and val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, name="best_model.pth")

        # Zapis końcowy
        self.save_checkpoint(num_epochs, name="final_model.pth", final=True)
        print("Trening zakończony.")
        return results

    def save_checkpoint(self, epoch, name=None, final=False):
        """Pomocnicza funkcja do zapisu."""
        if name is None:
            name = f"checkpoint_ep{epoch}.pth"

        path = os.path.join(self.save_dir, name)

        # Zapisujemy stan oryginalnego modelu (bez wrappera treningowego, jeśli to możliwe)
        if hasattr(self.model, "original_model"):
            state_dict = self.model.original_model.state_dict()
        else:
            state_dict = self.model.state_dict()

        torch.save(state_dict, path)
        if final:
            print(f"Zapisano końcowy model do: {path}")
