import argparse
from pathlib import Path

import matplotlib

#Save figures without needing graphical display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_test_patients(result_dir, method):
    "Load patient-lvl test predictions for selected method."

    if method == "linear":
        data = pd.read_csv(
            result_dir / "predictions.csv",
            dtype={"patient_id": str},
        )

        #File includes all splits, use only test samples.
        test_mask = (
            data["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["test", "testing"])
        )

        data = data.loc[test_mask].copy()

    else:
        #Fine-tuning already saves patient-lvl test predictions. 
        data = pd.read_csv(
            result_dir / "patient_predictions.csv",
            dtype={"PatientID": str},
        )

        data = data.rename(columns={"PatientID": "patient_id"})

    if data.empty:
        raise ValueError("No test predictions found.")

    if data["patient_id"].isna().any():
        raise ValueError("Missing patient IDs.")

    data["label"] = pd.to_numeric(data["label"], errors="raise")
    data["probability_lv"] = pd.to_numeric(
        data["probability_lv"],
        errors="raise",
    )

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be RV=0 or LV=1.")

    probabilities = data["probability_lv"]

    if (
        not np.isfinite(probabilities).all()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError("LV probabilities must be between 0 and 1.")

    if (data.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("A patient has conflicting labels.")

    if method == "linear":
        #Match existing pipeline's patient-lvl aggregation
        data = (
            data.groupby("patient_id")
            .agg(
                label=("label", "first"),
                probability_lv=("probability_lv", "mean"),
            )
            .reset_index()
        )

    elif data["patient_id"].duplicated().any():
        raise ValueError("Duplicate patients in patient_predictions.csv.")

    return data


def plot_confusion_matrix(result_dir, method):
    "Save a patient-lvl test confusion matrix."

    data = load_test_patients(result_dir, method)

    #Probability refers to LV: RV=0, LV=1
    predicted = (data["probability_lv"] >= 0.5).astype(int)

    matrix = confusion_matrix(
        data["label"],
        predicted,
        labels=[0, 1],
    )

    accuracy = (predicted == data["label"]).mean()
    name = "Linear probing" if method == "linear" else "Fine-tuning"

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["RV", "LV"],
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        f"{name} - patient-level test results\n"
        f"Accuracy: {accuracy:.1%}; patients: {len(data)}"
    )

    try:
        for extension in ("png", "pdf"):
            output_path = result_dir / (
                f"test_confusion_matrix.{extension}"
            )

            fig.savefig(output_path, dpi=200)
            print(f"Saved: {output_path}")
    finally: 
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save an RV/LV test confusion matrix."
    )

    parser.add_argument(
        "result_dir",
        type=Path,
        help="Directory containing the saved predictions",
    )

    parser.add_argument(
        "--method",
        choices=["linear", "finetune"],
        required=True,
    )

    args = parser.parse_args()
    plot_confusion_matrix(args.result_dir, args.method)

#Run seperately for each method
#For linear probing:
# python ventricle_project/plot_confusion_matrix.py\
#   ventricle_project/results/linear_probe_rvlv\
#   --method linear

#For fine-tuning:
# python ventricle_project/plot_confusion_matrix.py\
#   ventricle_project/results/finetune_rvlv\
#   --method finetune 