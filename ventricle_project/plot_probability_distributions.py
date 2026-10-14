import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

#Could import from confusion matrix plots file, but want each file to be independant
def load_test_patients(results_dir, method):
    "Read patient-level test predictions directly from training outputs."

    results_dir = Path(results_dir)

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

    data["probability_lv"] = pd.to_numeric(
        data["probability_lv"], errors="raise"
    )

    if not data["label"].isin([0, 1]).all():
        raise ValueError("Labels must be RV=0 or LV=1.")

    probabilities = data["probability_lv"]

    if (
        not np.isfinite(probabilities).all()
        or not probabilities.between(0, 1).all()
    ):
        raise ValueError(
            "LV probabilities must be finite and between 0 and 1."
        )

    if (data.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("A patient has conflicting labels.")

    if method == "linear":
        #Match training pipeline's patient-level evaluation.
        data = (
            data.groupby("patient_id")
            .agg(
                label=("label", "first"),
                probability_lv=("probability_lv", "mean"),
            )
            .reset_index()
        )

    elif data["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient-level predictions.")

    return data

def compare_probability_distributions(
        linear_dir,
        finetune_dir,
        output_dir,
):
    linear = load_test_patients(
        linear_dir, "linear"
    ).set_index("patient_id").sort_index()

    finetune = load_test_patients(
        finetune_dir, "finetune"
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
        "Patient-level test LV probabilities, grouped by true class"
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
                (0, "RV", "tab:blue"),
                (1, "LV", "tab:orange"),
            ]:
                values = data.loc[
                    data["label"] == label_id,
                    "probability_lv",
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
                    "method": method,
                    "true_class": class_name,
                    "n_patients": len(values),
                    "min_probability_lv": values.min(),
                    "median_probability_lv": values.median(),
                    "mean_probability_lv": values.mean(),
                    "max_probability_lv": values.max(),
                    "percent_predicted_lv": ((values >= 0.5).mean()*100),
                })

            predicted = (data["probability_lv"] >= 0.5).astype(int)

            accuracy = (predicted == data["label"]).mean()

            auroc = (
                roc_auc_score(
                    data["label"],
                    data["probability_lv"],
                )
                if data["label"].nunique() == 2
                else np.nan
            )

            print(
                f"{method}: "
                f"patient-level test accuracy={accuracy:.3f}, "
                f"AUROC={auroc:.3f}, "
                f"predicted LV={predicted.sum()}/{len(data)}"
            )

            ax.axvline(
                0.5,
                color="black",
                linestyle="--",
                label="Threshold = 0.5",
            )

            ax.set_title(method)
            ax.set_xlabel("Predicted LV probability")
            ax.set_ylabel("Probability density")
            ax.set_xlim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            ax.legend()

        summary = pd.DataFrame(summary_rows)

        summary_path = (
            output_dir / "probability_distribution_summary.csv"
        )
        summary.to_csv(summary_path, index=False)

        print("\nProbability summary:")
        print(summary.to_string(index=False))
        print(f"\nSaved: {summary_path}")

        for extension in ("png", "pdf"):
            figure_path = output_dir / (
                f"probability_distributions.{extension}"
            )
            fig.savefig(figure_path, dpi=200)
            print(f"Saved: {figure_path}")

    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RV/LV probability distributions."
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

    args = parser.parse_args()

    compare_probability_distributions(
        args.linear_dir,
        args.finetune_dir,
        args.output_dir,
    )

#Run as 
# python ventricle_project/plot_probability_distributions.py --linear-dir ventricle_project/results/linear_probe_rvlv --finetune-dir ventricle_project/results/finetune_rvlv --output-dir ventricle_project/results/comparison