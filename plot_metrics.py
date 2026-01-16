import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metrics(csv_path="metrics.csv"):

    if not Path(csv_path).exists():
        print(f"Błąd: Plik {csv_path} nie istnieje.")
        return

    df = pd.read_csv(csv_path)

    metrics_per_epoch = df.groupby("epoch").mean()
    cols = metrics_per_epoch.columns

    train_loss_col = [c for c in cols if "train_loss_epoch" in c]
    val_loss_col = [c for c in cols if "val_loss" in c]
    train_acc_col = [c for c in cols if "train_acc" in c]
    val_acc_col = [c for c in cols if "val_acc" in c]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    if train_loss_col and val_loss_col:
        ax1.plot(
            metrics_per_epoch.index,
            metrics_per_epoch[train_loss_col[0]],
            label="Train Loss",
            marker="o",
        )
        ax1.plot(
            metrics_per_epoch.index,
            metrics_per_epoch[val_loss_col[0]],
            label="Val Loss",
            marker="s",
        )
        ax1.set_title("Loss vs Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

    # Accuracy
    if train_acc_col and val_acc_col:
        ax2.plot(
            metrics_per_epoch.index,
            metrics_per_epoch[train_acc_col[0]],
            label="Train Acc",
            marker="o",
        )
        ax2.plot(
            metrics_per_epoch.index,
            metrics_per_epoch[val_acc_col[0]],
            label="Val Acc",
            marker="s",
        )
        ax2.set_title("Accuracy vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()


def get_test_metricies(csv_path="metrics.csv"):
    if not Path(csv_path).exists():
        print(f"Błąd: Plik {csv_path} nie istnieje.")
        return

    df = pd.read_csv(csv_path)

    test_results = df.dropna(subset=["test_Rank1"])

    cols_of_interest = ["test_mAP", "test_Rank1", "test_Rank5", "test_Rank10"]
    test_metricies = test_results[cols_of_interest]

    return list(test_metricies.values[0])
