import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_figure(fig, output_dir, name):
    "Save both an image and a PDF."
    try:
        for extension in ("png", "pdf"):
            path = output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(f"Saved: {path}")
    finally:
        plt.close(fig)

def validate_predictions(data):
    "Check labels and probabilities before calculating results."
    data = data.copy()
    data["label"] = pd.to_numeric(data["label"], errors="raise")
    data["probability_lv"] = pd.to_numeric(
        data["probability_lv"], errors="raise"
    )

    if data.empty:
        raise ValueError("No test predictions found.")

    if data["patient_id"].isna().any():
        raise ValueError("Missing patient IDs.")

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be RV=0 or LV=1.")

    probabilities = data["probability_lv"]

    if (
        not np.isfinite(probabilities).all()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError("LV probabilities must be finite and between 0 and 1.")

    return data

def load_linear_test_patients(result_dir):
    data = pd.read_csv(
        result_dir / "predictions.csv",
        dtype={"patient_id": str},
    )

    #Linear probing saves train, validation, and test predictions
    test_mask = (
        data["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["test", "testing"])
    )

    data = validate_predictions(data.loc[test_mask])

    #Every series from patient must havae same target label
    if (data.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("A patient has conflicting RV/LV labels.")

    return (
        data.groupby("patient_id")
        .agg(
            label=("label", "first"),
            probability_lv=("probability_lv", "mean"),
        )
        .sort_index()
    )


def load_finetune_test_patients(result_dir):
    #Existing fine-tuning script writes this from test predictions
    data = pd.read_csv(
        result_dir / "patient_predictions.csv",
        dtype={"PatientID": str},
    )

    data = data.rename(columns={"PatientID": "patient_id"})
    data = validate_predictions(data)

    if data["patient_id"].duplicated().any():
        raise ValueError("Duplicate patients in fine-tuning patient predictions.")

    return data.set_index("patient_id").sort_index()

#THE CONFUSION MATRICES
def plot_confusion_matrices(linear_dir, finetune_dir, output_dir):
    linear = load_linear_test_patients(linear_dir)
    finetune = load_finetune_test_patients(finetune_dir)

    #Ensure comparison of the same held-out patients.
    if set(linear.index) != set(finetune.index):
        raise ValueError(
            "The methods have different test patient IDs. "
            "Use matching test splits before comparing them. "
        )

    finetune = finetune.loc[linear.index]

    if not np.array_equal(
        linear["label"].to_numpy(),
        finetune["label"].to_numpy(),
    ):
        raise ValueError("The methods have different labels for test patients.")

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.5), constrained_layout=True
    )

    fig.suptitle(
        "Patient-level test classification - LV threshold = 0.5"
    )

    for ax, name, data in [
        (axes[0], "Linear probing", linear),
        (axes[1], "Fine-tuning", finetune),
    ]:
        predicted = (data["probability_lv"] >= 0.5).astype(int)

        matrix = confusion_matrix(
            data["label"],
            predicted,
            labels=[0, 1],
        )

        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=["RV", "LV"],
        ).plot(
            ax=ax,
            cmap="Blues",
            values_format="d",
            colorbar=False,
        )

        accuracy = (predicted == data["label"]).mean()
        ax.set_title(f"{name}\nTest accuracy: {accuracy:.1%}")

    save_figure(fig, output_dir, "test_confusion_matrices")


def plot_accuracy_comparison(linear_dir, finetune_dir, output_dir):
    histories = {
        "Linear probing": pd.read_csv(
            linear_dir / "training_history.csv"
        ),
        "Fine-tuning": pd.read_csv(
            finetune_dir / "training_history.csv"
        ),
    }

    required = ["epoch", "train_accuracy", "val_accuracy"]

    for name, history in histories.items():
        missing = [column for column in required if column not in history]

        if missing or history.empty:
            raise ValueError(
                f"{name}: missing columns {missing} or empty history. "
                "Use the updated training scripts."
            )

        for column in ("train_accuracy", "val_accuracy"):
            if not history[column].between(0, 1).all():
                raise ValueError(f"{name}: invalid values in {column}.")

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )

    fig.suptitle("Method comparison - series-lebel accuracy")

    for ax, column, title in [
        (axes[0], "train_accuracy", "Training"),
        (axes[1], "val_accuracy", "Validation"),
    ]:
        for name, history in histories.items():
            history = history.sort_values("epoch")

            ax.plot(
                history["epoch"],
                history[column],
                label=name,
                marker="o",
                markersize=3,
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend()

    save_figure(fig, output_dir, "accuracy_comparison")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RV/LV linear probing and fine-tuning."
    )

    parser.add_argument("--linear-dir", type=Path, required=True)
    parser.add_argument("--finetune-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ventricle_project/results/comparison"),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    #Uncomment to include confusion matrices in this file instead of using seperate file. 
    #plot_confusion_matrices(
    #    args.linear_dir, args.finetune-dir, args.output_dir
    #)

    plot_accuracy_comparison(
        args.linear_dir, args.finetune_dir, args.output_dir
    )

#Run it as:
# python ventricle_project/compare_rvlv.py\
#   --linear-dir ventricle_project/results/linear_probe_rvlv\
#   --finetune-dir ventricle_project/results/finetune_rvlv\ 
#   --output-dir ventricle_project/results/comparison
