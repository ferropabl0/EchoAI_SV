import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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

#Could import from confusion matrix plots file, but want each file to be independant
def load_test_patients(results_dir, method, task_config):
    "Read patient-level test predictions directly from training outputs."

    results_dir = Path(results_dir)
    probability_column = task_config["probability_column"]

    if method == "linear":
        data = pd.read_csv(
            results_dir / "predictions.csv",
            dtype={"patient_id": str},
        )

        #Linear probing saves predictions for all splits
        test_mask = (
            data["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["test", "testing"])
        )

        data = data.loc[test_mask].copy()

    elif method == "finetune":
        #Fine-tuning already saves patient-level test predictions.
        data = pd.read_csv(
            results_dir / "patient_predictions.csv",
            dtype={"PatientID": str},
        )

        data = data.rename(
            columns={"PatientID": "patient_id"}
        )

    else: 
        raise ValueError("Method must be linear or finetune.")

    if data.empty:
        raise ValueError("No test predictions found.")

    if data["patient_id"].isna().any():
        raise ValueError("Missing patient IDs.")

    data["label"] = pd.to_numeric(
        data["label"], errors="raise"
    )

    data[probability_column] = pd.to_numeric(
        data[probability_column], errors="raise"
    )

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
        raise ValueError(
            "Probabilities must be finite and between 0 and 1."
        )

    if (data.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("A patient has conflicting labels.")

    if method == "linear":
        #Match training pipeline's patient-level evaluation.
        data = (
            data.groupby("patient_id")
            .agg({
                "label": "first",
                probability_column: "mean",
            })
            .reset_index()
        )

    elif data["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient-level predictions.")

    return data

def compare_probability_distributions(
        linear_dir,
        finetune_dir,
        output_dir,
        task,
):

    task_config = TASK_CONFIGS[task]
    probability_column = task_config["probability_column"]
    negative_class, positive_class = task_config["class_names"]

    output_dir = Path(output_dir)

    linear = load_test_patients(
        linear_dir, "linear", task_config
    ).set_index("patient_id").sort_index()

    finetune = load_test_patients(
        finetune_dir, "finetune", task_config
    ).set_index("patient_id").sort_index()

    #Comparing patients with same clinical labels
    if set(linear.index) != set(finetune.index):
        raise ValueError(
            "Different test patietns between methods."
            "Use matching test splits."
        )

    #Align patients, then confirm their clinical labels agree.
    finetune = finetune.loc[linear.index]

    if not linear["label"].eq(finetune["label"]).all():
        raise ValueError(
            "Test patient labels differ between methods."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    fig.suptitle(
        f"{task_config['title']}\n"
        "Patient-level test probabilities, grouped by true class"
    )

    #Use identical bins for both methods and classes
    bins = np.linspace(0, 1, 21)
    summary_rows = []

    try:
        for ax, method, data in [
            (axes[0], "Linear probing", linear),
            (axes[1], "Fine-tuning", finetune),
        ]:
            for label_id, class_name, color in [
                (0, negative_class, "tab:blue"),
                (1, positive_class, "tab:orange"),
            ]:
                values = data.loc[
                    data["label"] == label_id,
                    probability_column,
                ]

                if values.empty:
                    print(f"{method}: no true {class_name} patients.")
                    continue

                ax.hist(
                    values,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    color=color,
                    label=f"True {class_name} (n={len(values)})",
                )

                summary_rows.append({
                    "task": task,
                    "method": method,
                    "true_class": class_name,
                    "positive_class": positive_class,
                    "n_patients": len(values),
                    "min_probability": values.min(),
                    "median_probability": values.median(),
                    "mean_probability": values.mean(),
                    "max_probability": values.max(),
                    "percent_predicted_positive": ((values >= 0.5).mean()*100),
                })

            predicted = (data[probability_column] >= 0.5).astype(int)

            accuracy = (predicted == data["label"]).mean()

            auroc = (
                roc_auc_score(
                    data["label"],
                    data[probability_column],
                )
                if data["label"].nunique() == 2
                else np.nan
            )

            print(
                f"{method}: "
                f"patient-level test accuracy={accuracy:.3f}, "
                f"AUROC={auroc:.3f}, "
                f"predicted {positive_class}="
                f"{predicted.sum()}/{len(data)}"
            )

            ax.axvline(
                0.5,
                color="black",
                linestyle="--",
                label="Threshold = 0.5",
            )

            ax.set_title(method)
            ax.set_xlabel(task_config["probability_label"])
            ax.set_ylabel("Probability density")
            ax.set_xlim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            ax.legend()

        summary = pd.DataFrame(summary_rows)

        summary_path = output_dir / (
             f"probability_distribution_summary_{task}.csv"
        )
        summary.to_csv(summary_path, index=False)

        print("\nProbability summary:")
        print(summary.to_string(index=False))
        print(f"\nSaved: {summary_path}")

        for extension in ("png", "pdf"):
            figure_path = output_dir / (
                f"probability_distributions_{task}.{extension}"
            )
            fig.savefig(figure_path, dpi=200)
            print(f"Saved: {figure_path}")

    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare patient-level classification probabilities."
    )

    parser.add_argument(
        "--linear-dir", type=Path, required=True
    )
    parser.add_argument(
        "--finetune-dir", type=Path, required=True
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ventricle_project/results/comparison"),
    )

    parser.add_argument(
        "--task",
        choices=["rvlv", "function"],
        required=True,
    )

    args = parser.parse_args()

    compare_probability_distributions(
        args.linear_dir,
        args.finetune_dir,
        args.output_dir,
        args.task,
    )

#Run different methods and tasks as:

# RV/LV:
# python ventricle_project/plot_probability_distributions.py --linear-dir ventricle_project/results/linear_probe_rvlv --finetune-dir ventricle_project/results/finetune_rvlv --output-dir ventricle_project/results/comparison --task rvlv

# Normal/impaired function:
# python ventricle_project/plot_probability_distributions.py --linear-dir ventricle_project/results/function_linear_probe --finetune-dir ventricle_project/results/function_finetune --output-dir ventricle_project/results/function_comparison --task function