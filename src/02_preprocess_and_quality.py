from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


# draw figures to files only
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# folders and fixed analysis settings
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
AUDIT_RESULTS_DIR = PROJECT_DIR / "results" / "audit"
FIGURES_DIR = PROJECT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
NEAR_CONSTANT_SHARE = 0.98

PROCESS_COLORS = {"AlCu": "#2F6690", "WTi": "#C46D3B"}


# train and test splits


# factorizing the full rows keeps only exactly equal 17-value profiles together
def exact_profile_groups(targets):
    profile_index = pd.MultiIndex.from_frame(targets)
    group_numbers, _ = pd.factorize(profile_index, sort=False)
    return group_numbers


# label every row train, then mark the test rows
def split_labels(row_count, test_indices):
    labels = np.full(row_count, "train", dtype=object)
    labels[test_indices] = "test"
    return labels


# one row per wafer with its split labels, profile group and zero-output flag
def make_split_assignments(targets):
    observation_indices = np.arange(len(targets))

    # the primary validation is the requested ordinary fixed random split
    primary_train, primary_test = train_test_split(
        observation_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    primary_labels = split_labels(len(targets), primary_test)

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
    grouped_labels = split_labels(len(targets), grouped_test)

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


# feature screening


# map each exact duplicate column to the first earlier column it matches
def find_duplicate_columns(data):
    duplicates = {}
    columns = list(data.columns)
    for column_number, column in enumerate(columns):
        for earlier_column in columns[:column_number]:
            if data[column].equals(data[earlier_column]):
                duplicates[column] = earlier_column
                break
    return duplicates


# the first matching rule gives the reason, in this order
def screening_reason(is_constant, is_near_constant, duplicate_of):
    if is_constant:
        return "constant in primary training data"
    if is_near_constant:
        return "one value occurs in at least 98% of primary training rows"
    if duplicate_of:
        return f"exact duplicate of {duplicate_of} in primary training data"
    return "retained"


# feature decisions use training rows only so the test rows remain untouched
def screen_features(process_inputs, assignments):
    train_mask = assignments["primary_split"].eq("train").to_numpy()
    training_inputs = process_inputs.loc[train_mask]
    duplicate_columns = find_duplicate_columns(training_inputs)
    rows = []

    for column in training_inputs.columns:
        value_counts = training_inputs[column].value_counts(dropna=False)
        modal_share = float(value_counts.iloc[0] / len(training_inputs))
        unique_values = training_inputs[column].nunique(dropna=False)
        is_constant = unique_values <= 1
        is_near_constant = modal_share >= NEAR_CONSTANT_SHARE
        duplicate_of = duplicate_columns.get(column, "")
        keep = not (is_constant or is_near_constant or bool(duplicate_of))

        rows.append(
            {
                "feature": column,
                "training_unique_values": int(unique_values),
                "training_modal_share": modal_share,
                "training_zero_share": float((training_inputs[column] == 0).mean()),
                "keep_for_modeling": keep,
                "decision_reason": screening_reason(is_constant, is_near_constant, duplicate_of),
            }
        )

    return pd.DataFrame(rows)


# quality metrics from the released targets


# per-wafer summaries of the 17 released targets, plus the reference profile
# the reference profile uses valid primary training rows only
def calculate_quality_metrics(targets, assignments):
    is_train = assignments["primary_split"].eq("train")
    is_valid = ~assignments["all_zero_output"]
    reference_mask = is_train & is_valid
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


# distribution of each metric over wafers whose targets are not all zero
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


# row counts and profile overlap between train and test for both splits
def summarize_splits(process, assignments):
    rows = []
    split_names = [
        ("primary_split", "ordinary fixed split"),
        ("grouped_sensitivity_split", "identical-target-profile sensitivity"),
    ]
    for split_column, validation_name in split_names:
        train = assignments[split_column].eq("train")
        test = assignments[split_column].eq("test")
        train_groups = set(assignments.loc[train, "target_profile_group"])
        test_groups = set(assignments.loc[test, "target_profile_group"])
        test_profiles_seen = assignments.loc[test, "target_profile_group"].isin(train_groups)
        zero_output = assignments["all_zero_output"]
        rows.append(
            {
                "process": process,
                "validation": validation_name,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "test_share": float(test.mean()),
                "target_profile_groups_in_both_sets": len(train_groups & test_groups),
                "test_rows_with_profile_seen_in_train": int(test_profiles_seen.sum()),
                "zero_output_rows_in_train": int((train & zero_output).sum()),
                "zero_output_rows_in_test": int((test & zero_output).sum()),
            }
        )
    return rows


# figure


# one histogram panel in the shared style
def plot_histogram(axis, values, color, title, x_label):
    axis.hist(
        values,
        bins=32,
        color=color,
        edgecolor="white",
        linewidth=0.4,
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Wafers")


# one row per process: mean index on the left, shape deviation on the right
def plot_quality_distributions(process_metrics):
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))

    for row_number, process in enumerate(["AlCu", "WTi"]):
        metrics = process_metrics[process]
        valid = metrics.loc[~metrics["all_zero_output"]]
        color = PROCESS_COLORS[process]

        plot_histogram(
            axes[row_number, 0],
            valid["released_mean_index"],
            color,
            f"{process}: Released-Scale Mean Index",
            "Mean of 17 released target values",
        )
        plot_histogram(
            axes[row_number, 1],
            valid["reference_profile_shape_deviation"],
            color,
            f"{process}: Reference-Profile Shape Deviation",
            "Dimensionless RMS deviation",
        )

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


# running the step


# write the four processed tables for one process
def save_process_tables(process, assignments, feature_screening, metrics, reference_profile):
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
    reference_table = pd.DataFrame(
        {
            "target": reference_profile.index,
            "primary_training_median": reference_profile.values,
        }
    )
    reference_table.to_csv(
        PROCESSED_DIR / f"{process_name}_reference_profile.csv", index=False
    )


# say how many features were kept and which were dropped
def print_screening_result(process, feature_screening):
    removed_features = feature_screening.loc[
        ~feature_screening["keep_for_modeling"], "feature"
    ].tolist()
    print(
        f"{process}: retained {feature_screening['keep_for_modeling'].sum()} of "
        f"{len(feature_screening)} features; excluded {removed_features}."
    )


# build splits, screening and quality metrics for both processes, then summarize
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

        save_process_tables(process, assignments, feature_screening, metrics, reference_profile)

        quality_summary_rows.extend(summarize_metrics(process, metrics))
        split_summary_rows.extend(summarize_splits(process, assignments))
        process_metrics[process] = metrics

        print_screening_result(process, feature_screening)

    quality_summary = pd.DataFrame(quality_summary_rows)
    split_summary = pd.DataFrame(split_summary_rows)
    quality_summary.to_csv(AUDIT_RESULTS_DIR / "quality_metric_summary.csv", index=False)
    split_summary.to_csv(AUDIT_RESULTS_DIR / "split_summary.csv", index=False)
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
