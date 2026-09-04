from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
NEAR_CONSTANT_SHARE = 0.98


def exact_profile_groups(targets):
    # factorizing the full rows keeps only exactly equal 17-value profiles together
    profile_index = pd.MultiIndex.from_frame(targets)
    group_numbers, _ = pd.factorize(profile_index, sort=False)
    return group_numbers


def make_split_assignments(targets):
    observation_indices = np.arange(len(targets))

    # the primary validation is the requested ordinary fixed random split
    primary_train, primary_test = train_test_split(
        observation_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    primary_labels = np.full(len(targets), "train", dtype=object)
    primary_labels[primary_test] = "test"

    # the grouped split is only a sensitivity check for repeated target profiles
    group_numbers = exact_profile_groups(targets)
    grouped_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    grouped_train, grouped_test = next(
        grouped_splitter.split(observation_indices, groups=group_numbers)
    )

    grouped_labels = np.full(len(targets), "train", dtype=object)
    grouped_labels[grouped_test] = "test"

    all_zero_output = (targets == 0).all(axis=1).to_numpy()
    assignments = pd.DataFrame(
        {
            "observation_index": observation_indices,
            "primary_split": primary_labels,
            "grouped_sensitivity_split": grouped_labels,
            "target_profile_group": group_numbers,
            "all_zero_output": all_zero_output,
        }
    )
    return assignments


def find_duplicate_columns(data):
    duplicates = {}
    columns = list(data.columns)
    for column_number, column in enumerate(columns):
        for earlier_column in columns[:column_number]:
            if data[column].equals(data[earlier_column]):
                duplicates[column] = earlier_column
                break
    return duplicates


def screen_features(process_inputs, assignments):
    # feature decisions use training rows only so the test rows remain untouched
    train_mask = assignments["primary_split"].eq("train").to_numpy()
    training_inputs = process_inputs.loc[train_mask]
    duplicate_columns = find_duplicate_columns(training_inputs)
    rows = []

    for column in training_inputs.columns:
        value_counts = training_inputs[column].value_counts(dropna=False)
        modal_share = float(value_counts.iloc[0] / len(training_inputs))
        is_constant = training_inputs[column].nunique(dropna=False) <= 1
        is_near_constant = modal_share >= NEAR_CONSTANT_SHARE
        duplicate_of = duplicate_columns.get(column, "")

        if is_constant:
            reason = "constant in primary training data"
        elif is_near_constant:
            reason = "one value occurs in at least 98% of primary training rows"
        elif duplicate_of:
            reason = f"exact duplicate of {duplicate_of} in primary training data"
        else:
            reason = "retained"

        rows.append(
            {
                "feature": column,
                "training_unique_values": int(training_inputs[column].nunique(dropna=False)),
                "training_modal_share": modal_share,
                "training_zero_share": float((training_inputs[column] == 0).mean()),
                "keep_for_modeling": not (is_constant or is_near_constant or bool(duplicate_of)),
                "decision_reason": reason,
            }
        )

    return pd.DataFrame(rows)


def calculate_quality_metrics(targets, assignments):
    # the reference profile uses valid primary training rows only
    reference_mask = assignments["primary_split"].eq("train") & ~assignments[
        "all_zero_output"
    ]
    reference_profile = targets.loc[reference_mask.to_numpy()].median(axis=0)

    released_mean = targets.mean(axis=1)
    released_sd = targets.std(axis=1, ddof=1)
    released_range = targets.max(axis=1) - targets.min(axis=1)
    released_cv = released_sd.divide(released_mean.replace(0, np.nan))

    # divide out the point baselines, then remove each row level to compare shape only
    point_relative = targets.divide(reference_profile, axis=1)
    row_relative_mean = point_relative.mean(axis=1).replace(0, np.nan)
    shape_relative = point_relative.divide(row_relative_mean, axis=0)
    profile_shape_deviation = np.sqrt(((shape_relative - 1) ** 2).mean(axis=1))

    metrics = assignments.copy()
    metrics["released_mean_index"] = released_mean
    metrics["released_standard_deviation"] = released_sd
    metrics["released_range"] = released_range
    metrics["released_coefficient_of_variation"] = released_cv
    metrics["reference_profile_shape_deviation"] = profile_shape_deviation
    return metrics, reference_profile


def summarize_metrics(process, metrics):
    valid_metrics = metrics.loc[~metrics["all_zero_output"]]
    metric_columns = [
        "released_mean_index",
        "released_standard_deviation",
        "released_range",
        "released_coefficient_of_variation",
        "reference_profile_shape_deviation",
    ]
    rows = []

    for metric in metric_columns:
        values = valid_metrics[metric].dropna()
        rows.append(
            {
                "process": process,
                "metric": metric,
                "observations": len(values),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "maximum": float(values.max()),
            }
        )
    return rows


def summarize_splits(process, assignments):
    rows = []
    for split_column, validation_name in [
        ("primary_split", "ordinary fixed split"),
        ("grouped_sensitivity_split", "identical-target-profile sensitivity"),
    ]:
        train = assignments[split_column].eq("train")
        test = assignments[split_column].eq("test")
        train_groups = set(assignments.loc[train, "target_profile_group"])
        test_groups = set(assignments.loc[test, "target_profile_group"])
        rows.append(
            {
                "process": process,
                "validation": validation_name,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "test_share": float(test.mean()),
                "target_profile_groups_in_both_sets": len(train_groups & test_groups),
                "test_rows_with_profile_seen_in_train": int(
                    assignments.loc[test, "target_profile_group"].isin(train_groups).sum()
                ),
                "zero_output_rows_in_train": int(
                    (train & assignments["all_zero_output"]).sum()
                ),
                "zero_output_rows_in_test": int(
                    (test & assignments["all_zero_output"]).sum()
                ),
            }
        )
    return rows


def plot_quality_distributions(process_metrics):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {"AlCu": "#2F6690", "WTi": "#C46D3B"}

    for row_number, process in enumerate(["AlCu", "WTi"]):
        metrics = process_metrics[process]
        valid = metrics.loc[~metrics["all_zero_output"]]

        axes[row_number, 0].hist(
            valid["released_mean_index"],
            bins=32,
            color=colors[process],
            edgecolor="white",
            linewidth=0.4,
        )
        axes[row_number, 0].set_title(f"{process}: Released-Scale Mean Index")
        axes[row_number, 0].set_xlabel("Mean of 17 released target values")
        axes[row_number, 0].set_ylabel("Wafers")

        axes[row_number, 1].hist(
            valid["reference_profile_shape_deviation"],
            bins=32,
            color=colors[process],
            edgecolor="white",
            linewidth=0.4,
        )
        axes[row_number, 1].set_title(f"{process}: Reference-Profile Shape Deviation")
        axes[row_number, 1].set_xlabel("Dimensionless RMS deviation")
        axes[row_number, 1].set_ylabel("Wafers")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Released Output Summaries (All-Zero Records Excluded)",
        fontsize=14,
        y=1.01,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "01_released_output_summaries.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    quality_summary_rows = []
    split_summary_rows = []
    process_metrics = {}

    for process in ["AlCu", "WTi"]:
        process_inputs = pd.read_csv(RAW_DIR / f"X_pvd_{process}.csv")
        targets = pd.read_csv(RAW_DIR / f"Y_pvd_{process}.csv")

        assignments = make_split_assignments(targets)
        feature_screening = screen_features(process_inputs, assignments)
        metrics, reference_profile = calculate_quality_metrics(targets, assignments)

        process_name = process.lower()
        assignments.to_csv(
            PROCESSED_DIR / f"{process_name}_split_assignments.csv", index=False
        )
        feature_screening.to_csv(
            PROCESSED_DIR / f"{process_name}_feature_screening.csv", index=False
        )
        metrics.to_csv(
            PROCESSED_DIR / f"{process_name}_quality_metrics.csv", index=False
        )
        pd.DataFrame(
            {
                "target": reference_profile.index,
                "primary_training_median": reference_profile.values,
            }
        ).to_csv(
            PROCESSED_DIR / f"{process_name}_reference_profile.csv", index=False
        )

        quality_summary_rows.extend(summarize_metrics(process, metrics))
        split_summary_rows.extend(summarize_splits(process, assignments))
        process_metrics[process] = metrics

        removed_features = feature_screening.loc[
            ~feature_screening["keep_for_modeling"], "feature"
        ].tolist()
        print(
            f"{process}: retained {feature_screening['keep_for_modeling'].sum()} of "
            f"{len(feature_screening)} features; excluded {removed_features}."
        )

    quality_summary = pd.DataFrame(quality_summary_rows)
    split_summary = pd.DataFrame(split_summary_rows)
    quality_summary.to_csv(RESULTS_DIR / "quality_metric_summary.csv", index=False)
    split_summary.to_csv(RESULTS_DIR / "split_summary.csv", index=False)
    plot_quality_distributions(process_metrics)

    print()
    print("Split summary:")
    print(split_summary.to_string(index=False))
    print()
    print("Quality metric summary:")
    print(
        quality_summary[
            ["process", "metric", "observations", "mean", "median", "maximum"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
