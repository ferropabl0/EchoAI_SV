import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

TASK_CONFIGS = {
    "rvlv": {"title": "RV/LV classification"},
    "function": {"title": "Normal/impaired ventricular function"},
}

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

def plot_accuracy_comparison(linear_dir, finetune_dir, output_dir, task):
    task_config = TASK_CONFIGS[task]
    linear_dir = Path(linear_dir)
    finetune_dir = Path(finetune_dir)
    output_dir = Path(output_dir)
    
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
            )

        for column in required:
            history[column] = pd.to_numeric(
                history[column], errors="raise"
            )

        if not np.isfinite(history[required].to_numpy()).all():
                raise ValueError(f"{name}: history contains missing/non-finite values.")

        if history["epoch"].duplicated().any():
            raise ValueError(
                f"{name}: invalid values in {column}."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), constrained_layout=True
    )

    fig.suptitle(
        f"{task_config['title']}\n"
        "Method comparison - series-level accuracy"
    )

    try:
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

        for extension in ("png", "pdf"):
            path = output_dir / (
                f"accuracy_comparison_{task}.{extension}"
            )

            fig.savefig(
                path, dpi=200, bbox_inches="tight"
            )
            print(f"Saved: {path}")

    finally:
        plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare training and validation accuracy between methods."
    )

    parser.add_argument("--linear-dir", type=Path, required=True)
    parser.add_argument("--finetune-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=list(TASK_CONFIGS),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ventricle_project/results/comparison"),
    )

    args = parser.parse_args()

    plot_accuracy_comparison(
        args.linear_dir, args.finetune_dir, args.output_dir, args.task,
    )

# Run as:
#python ventricle_project/accuracy_comparison.py --linear-dir ventricle_project/results/function_linear_probe --finetune-dir ventricle_project/results/function_finetune --output-dir ventricle_project/results/function_comparison --task function

#python ventricle_project/accuracy_comparison.py --linear-dir ventricle_project/results/linear_probe_rvlv --finetune-dir ventricle_project/results/finetune_rvlv --output-dir ventricle_project/results/comparison --task rvlv