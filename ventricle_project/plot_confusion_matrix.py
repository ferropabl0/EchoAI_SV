import argparse
from pathlib import Path

import matplotlib

#Save figures without needing graphical display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Determening if it's used for RV/LV or NORMAL/IMPAIRED
TASK_CONFIGS = {
    "rvlv": {
        "class_names": ["RV", "LV"],
        "probability_column": "probability_lv",
        "probability_label": "Predicted LV probability",
        "title": "RV/LV classification",
    },
    "function": {
        "class_names": ["Normal", "Impaired"],
        "probability_column": "probability_impaired",
        "probability_label": "Predicted impaired-function probability",
        "title": "Ventricular function classification",
    }
}

def load_test_patients(result_dir, method, task_config):
    "Load patient-level test predictions for selected method and task."

    result_dir = Path(result_dir)
    probability_column = task_config["probability_column"]

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
        #Fine-tuning already saves patient-level test predictions. 
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

    data[probability_column] = pd.to_numeric(data[probability_column], errors="raise")

    if not data["label"].isin([0, 1]).all():
        negative_class, positive_class = task_config["class_names"]
        raise ValueError(
            f"Labels must be {negative_class}=0 "
            f"or {positive_class}=1."
        )

    probabilities = data[probability_column]

    if (
        not np.isfinite(probabilities).all()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError("Probabilites must be finite and between 0 and 1.")

    if (data.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("A patient has conflicting labels.")

    if method == "linear":
        #Match existing pipeline's patient-level aggregation
        data = (
            data.groupby("patient_id")
            .agg({
                "label": "first",
                probability_column: "mean",
            })
            .reset_index()
        )

    elif data["patient_id"].duplicated().any():
        raise ValueError("Duplicate patients in patient_predictions.csv.")

    return data


def plot_confusion_matrix(result_dir, method, task_config):
    "Save a patient-level test confusion matrix."

    result_dir = Path(result_dir)
    probability_column = task_config["probability_column"]

    data = load_test_patients(result_dir, method, task_config)

    #Probability refers to LV or IMPAIRED.
    predicted = (data[probability_column] >= 0.5).astype(int)

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
        display_labels=task_config["class_names"],
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title(
        f"{task_config['title']}\n"
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
        description="Save a binary-classification test confusion matrix."
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

    parser.add_argument(
            "--task",
            choices=["rvlv", "function"],
            required=True,
        )
    
    args = parser.parse_args()
    
    task_config = TASK_CONFIGS[args.task]

    plot_confusion_matrix(args.result_dir, args.method, task_config)

#Running the different methods and configs:

# RV/LV linear probing:
# python ventricle_project/plot_confusion_matrix.py ventricle_project/results/linear_probe_rvlv --method linear --task rvlv

# RV/LV fine-tuning:
# python ventricle_project/plot_confusion_matrix.py ventricle_project/results/finetune_rvlv --method finetune --task rvlv

# Function linear probing:
# python ventricle_project/plot_confusion_matrix.py ventricle_project/results/function_linear_probe --method linear --task function

# Function fine-tuning:
# python ventricle_project/plot_confusion_matrix.py ventricle_project/results/function_finetune --method finetune --task function