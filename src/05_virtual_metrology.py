from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RANDOM_STATE = 42
CV_FOLDS = 5
RIDGE_ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
    10_000_000.0,
    100_000_000.0,
    1_000_000_000.0,
]
FOREST_SEEDS = [11, 42, 73]
FOREST_TREES = 300
FOREST_MINIMUM_LEAF_SIZE = 5
FOREST_MAX_FEATURE_SHARE = 0.70
FOREST_REQUIRED_RMSE_IMPROVEMENT = 0.10
CATEGORY_ORDER = ["normal", "t2 alarm only", "q alarm only", "both alarms"]
CATEGORY_COLORS = {
    "normal": "#8A94A6",
    "t2 alarm only": "#C44E52",
    "q alarm only": "#E39C37",
    "both alarms": "#7A3E9D",
}


def aggregate_rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def aggregate_mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def make_model(model):
    # scaling stays inside the pipeline so each cross-validation fold fits it alone
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def make_random_forest(seed):
    # trees do not need feature scaling and this fixed setting avoids a large search
    return RandomForestRegressor(
        n_estimators=FOREST_TREES,
        min_samples_leaf=FOREST_MINIMUM_LEAF_SIZE,
        max_features=FOREST_MAX_FEATURE_SHARE,
        random_state=seed,
        n_jobs=-1,
    )


def make_cv_folds(inputs, targets, groups=None):
    if groups is None:
        splitter = KFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        folds = list(splitter.split(inputs))
        cv_method = "shuffled 5-fold"
    else:
        splitter = GroupKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        folds = list(splitter.split(inputs, targets, groups))
        cv_method = "grouped 5-fold by exact target profile"
    return folds, cv_method


def choose_ridge_alpha(inputs, targets, groups=None):
    folds, cv_method = make_cv_folds(inputs, targets, groups)

    rows = []
    for alpha in RIDGE_ALPHAS:
        fold_errors = []
        for train_rows, validation_rows in folds:
            model = make_model(Ridge(alpha=alpha))
            model.fit(inputs[train_rows], targets[train_rows])
            predictions = model.predict(inputs[validation_rows])
            fold_errors.append(
                aggregate_rmse(targets[validation_rows], predictions)
            )

        rows.append(
            {
                "alpha": alpha,
                "cross_validation_method": cv_method,
                "folds": len(fold_errors),
                "mean_validation_rmse": float(np.mean(fold_errors)),
                "standard_deviation_validation_rmse": float(
                    np.std(fold_errors, ddof=1)
                ),
                "maximum_validation_rmse": float(np.max(fold_errors)),
            }
        )

    search_results = pd.DataFrame(rows)
    best_row = search_results.sort_values(
        ["mean_validation_rmse", "alpha"]
    ).iloc[0]
    return float(best_row["alpha"]), search_results


def evaluate_random_forest_gate(inputs, targets, ridge_cv_rmse, groups=None):
    # the optional model is admitted using training folds, never assessment rows
    folds, cv_method = make_cv_folds(inputs, targets, groups)
    rows = []
    for seed in FOREST_SEEDS:
        fold_errors = []
        for train_rows, validation_rows in folds:
            model = make_random_forest(seed)
            model.fit(inputs[train_rows], targets[train_rows])
            predictions = model.predict(inputs[validation_rows])
            fold_errors.append(
                aggregate_rmse(targets[validation_rows], predictions)
            )
        mean_error = float(np.mean(fold_errors))
        improvement = 1 - mean_error / ridge_cv_rmse
        rows.append(
            {
                "cross_validation_method": cv_method,
                "random_state": seed,
                "folds": len(fold_errors),
                "trees": FOREST_TREES,
                "minimum_leaf_size": FOREST_MINIMUM_LEAF_SIZE,
                "maximum_feature_share": FOREST_MAX_FEATURE_SHARE,
                "mean_validation_rmse": mean_error,
                "standard_deviation_validation_rmse": float(
                    np.std(fold_errors, ddof=1)
                ),
                "maximum_validation_rmse": float(np.max(fold_errors)),
                "ridge_mean_validation_rmse": ridge_cv_rmse,
                "rmse_improvement_over_ridge": improvement,
                "passes_ten_percent_gate": improvement
                >= FOREST_REQUIRED_RMSE_IMPROVEMENT,
            }
        )
    return pd.DataFrame(rows)


def reference_profile_shape(values, reference_profile):
    values = np.asarray(values, dtype=float)
    point_relative = values / reference_profile
    row_relative_mean = point_relative.mean(axis=1)
    valid_rows = np.isfinite(row_relative_mean) & ~np.isclose(row_relative_mean, 0)
    deviations = np.full(len(values), np.nan)
    shape_relative = point_relative[valid_rows] / row_relative_mean[valid_rows, None]
    deviations[valid_rows] = np.sqrt(np.mean((shape_relative - 1) ** 2, axis=1))
    return deviations


def summarize_model(
    validation,
    zero_treatment,
    model_name,
    alpha,
    train_mask,
    test_mask,
    assignments,
    actual,
    predicted,
    reference_profile,
):
    train_groups = set(assignments.loc[train_mask, "target_profile_group"])
    test_groups = set(assignments.loc[test_mask, "target_profile_group"])
    actual_mean = actual.mean(axis=1)
    predicted_mean = predicted.mean(axis=1)
    actual_shape = reference_profile_shape(actual, reference_profile)
    predicted_shape = reference_profile_shape(predicted, reference_profile)
    shape_rows = np.isfinite(actual_shape) & np.isfinite(predicted_shape)

    return {
        "validation": validation,
        "zero_output_treatment": zero_treatment,
        "model": model_name,
        "ridge_alpha": alpha,
        "training_rows": int(train_mask.sum()),
        "assessment_rows": int(test_mask.sum()),
        "target_profile_groups_in_both_sets": len(train_groups & test_groups),
        "aggregate_mae_17_targets": aggregate_mae(actual, predicted),
        "aggregate_rmse_17_targets": aggregate_rmse(actual, predicted),
        "aggregate_r2_17_targets": float(
            r2_score(actual, predicted, multioutput="uniform_average")
        ),
        "released_mean_index_mae": aggregate_mae(actual_mean, predicted_mean),
        "released_mean_index_rmse": aggregate_rmse(actual_mean, predicted_mean),
        "profile_shape_deviation_rows": int(shape_rows.sum()),
        "profile_shape_deviation_mae": aggregate_mae(
            actual_shape[shape_rows], predicted_shape[shape_rows]
        ),
        "profile_shape_deviation_rmse": aggregate_rmse(
            actual_shape[shape_rows], predicted_shape[shape_rows]
        ),
    }


def summarize_targets(validation, model_name, alpha, target_names, actual, predicted):
    rows = []
    for target_number, target in enumerate(target_names):
        target_actual = actual[:, target_number]
        target_predicted = predicted[:, target_number]
        rows.append(
            {
                "validation": validation,
                "zero_output_treatment": "excluded",
                "model": model_name,
                "ridge_alpha": alpha,
                "target": target,
                "mae": aggregate_mae(target_actual, target_predicted),
                "rmse": aggregate_rmse(target_actual, target_predicted),
                "r2": float(r2_score(target_actual, target_predicted)),
            }
        )
    return rows


def fit_and_assess(inputs, targets, train_mask, test_mask, model):
    fitted_model = make_model(model)
    fitted_model.fit(inputs[train_mask], targets[train_mask])
    predictions = fitted_model.predict(inputs[test_mask])
    return fitted_model, predictions


def build_prediction_table(
    assignments,
    monitoring,
    test_mask,
    target_names,
    actual,
    predicted,
    reference_profile,
):
    observation_indices = assignments.loc[test_mask, "observation_index"].to_numpy()
    states = monitoring.set_index("observation_index").loc[
        observation_indices,
        ["alarm_category", "t2_ratio", "q_ratio"],
    ]
    table = pd.DataFrame({"observation_index": observation_indices})
    table["alarm_category"] = states["alarm_category"].to_numpy()
    table["t2_ratio"] = states["t2_ratio"].to_numpy()
    table["q_ratio"] = states["q_ratio"].to_numpy()
    table["row_mae_17_targets"] = np.mean(np.abs(actual - predicted), axis=1)
    table["row_rmse_17_targets"] = np.sqrt(
        np.mean((actual - predicted) ** 2, axis=1)
    )
    table["actual_released_mean_index"] = actual.mean(axis=1)
    table["predicted_released_mean_index"] = predicted.mean(axis=1)
    table["actual_profile_shape_deviation"] = reference_profile_shape(
        actual, reference_profile
    )
    table["predicted_profile_shape_deviation"] = reference_profile_shape(
        predicted, reference_profile
    )

    for target_number, target in enumerate(target_names):
        table[f"actual_{target}"] = actual[:, target_number]
        table[f"predicted_{target}"] = predicted[:, target_number]
    return table


def summarize_error_by_state(prediction_table):
    rows = []
    for category in CATEGORY_ORDER:
        values = prediction_table.loc[
            prediction_table["alarm_category"].eq(category),
            "row_mae_17_targets",
        ]
        rows.append(
            {
                "alarm_category": category,
                "observations": len(values),
                "mean_row_mae": float(values.mean()) if len(values) else np.nan,
                "standard_deviation_row_mae": float(values.std(ddof=1))
                if len(values) > 1
                else np.nan,
                "median_row_mae": float(values.median()) if len(values) else np.nan,
                "p90_row_mae": float(values.quantile(0.90)) if len(values) else np.nan,
                "maximum_row_mae": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_primary_predictions(actual, predicted, prediction_table, primary_metrics):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    cells = axes[0].hexbin(
        actual.ravel(),
        predicted.ravel(),
        gridsize=58,
        mincnt=1,
        cmap="Blues",
    )
    axes[0].plot([lower, upper], [lower, upper], color="#C44E52", linestyle="--")
    axes[0].set_xlabel("Actual released target value")
    axes[0].set_ylabel("Predicted released target value")
    axes[0].set_title("All 17 Direct Targets")
    figure.colorbar(cells, ax=axes[0], label="Assessment cells per hexagon")

    for category in CATEGORY_ORDER:
        group = prediction_table.loc[
            prediction_table["alarm_category"].eq(category)
        ]
        axes[1].scatter(
            group["actual_released_mean_index"],
            group["predicted_released_mean_index"],
            s=18,
            alpha=0.55 if category == "normal" else 0.90,
            color=CATEGORY_COLORS[category],
            label=f"{category} (n={len(group)})",
        )
    mean_lower = float(
        min(
            prediction_table["actual_released_mean_index"].min(),
            prediction_table["predicted_released_mean_index"].min(),
        )
    )
    mean_upper = float(
        max(
            prediction_table["actual_released_mean_index"].max(),
            prediction_table["predicted_released_mean_index"].max(),
        )
    )
    axes[1].plot(
        [mean_lower, mean_upper],
        [mean_lower, mean_upper],
        color="#C44E52",
        linestyle="--",
    )
    axes[1].set_xlabel("Actual released-scale mean index")
    axes[1].set_ylabel("Predicted released-scale mean index")
    axes[1].set_title("Secondary Mean-Index Summary")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "AlCu Random Forest Virtual Metrology on the Ordinary Held-Out Split\n"
        f"17-target RMSE = {primary_metrics['aggregate_rmse_17_targets']:.4f}; "
        f"aggregate R² = {primary_metrics['aggregate_r2_17_targets']:.3f}",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "08_alcu_virtual_metrology_predictions.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_per_target_errors(per_target_metrics):
    primary = per_target_metrics.loc[
        per_target_metrics["validation"].eq("ordinary fixed split")
    ].copy()
    primary["target_number"] = primary["target"].str.extract(r"(\d+)$").astype(int)
    linear = primary.loc[primary["model"].eq("Linear Regression")].sort_values(
        "target_number"
    )
    ridge = primary.loc[primary["model"].eq("Ridge")].sort_values("target_number")
    forest = primary.loc[primary["model"].eq("Random Forest")].sort_values(
        "target_number"
    )
    positions = np.arange(1, len(linear) + 1)
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        positions - 0.25,
        linear["mae"],
        width=0.25,
        color="#8A94A6",
        label="Linear Regression",
    )
    axis.bar(
        positions,
        ridge["mae"],
        width=0.25,
        color="#2F6690",
        label="Ridge",
    )
    axis.bar(
        positions + 0.25,
        forest["mae"],
        width=0.25,
        color="#4C956C",
        label="Random Forest",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(positions)
    axis.set_xlabel("Released target measurement number")
    axis.set_ylabel("Mean absolute error")
    axis.set_title("AlCu Per-Target Error on the Ordinary Held-Out Split")
    axis.grid(axis="y", alpha=0.20)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "09_alcu_per_target_errors.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_error_by_state(prediction_table):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    state_values = [
        prediction_table.loc[
            prediction_table["alarm_category"].eq(category),
            "row_mae_17_targets",
        ].to_numpy()
        for category in CATEGORY_ORDER
    ]
    boxes = axes[0].boxplot(
        state_values,
        tick_labels=CATEGORY_ORDER,
        patch_artist=True,
        showfliers=True,
    )
    for box, category in zip(boxes["boxes"], CATEGORY_ORDER):
        box.set_facecolor(CATEGORY_COLORS[category])
        box.set_alpha(0.65)
    axes[0].set_xlabel("Retrospective process state")
    axes[0].set_ylabel("Row MAE across 17 targets")
    axes[0].set_title("Prediction Error by Empirical Alert Category")
    axes[0].tick_params(axis="x", rotation=20)

    maximum_ratio = prediction_table[["t2_ratio", "q_ratio"]].max(axis=1)
    for category in CATEGORY_ORDER:
        rows = prediction_table["alarm_category"].eq(category)
        axes[1].scatter(
            maximum_ratio[rows],
            prediction_table.loc[rows, "row_mae_17_targets"],
            s=18,
            alpha=0.55 if category == "normal" else 0.90,
            color=CATEGORY_COLORS[category],
            label=category,
        )
    axes[1].axvline(1, color="#444444", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Maximum of T² and Q threshold ratios")
    axes[1].set_ylabel("Row MAE across 17 targets")
    axes[1].set_title("Prediction Error versus Process-State Distance")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "AlCu Random Forest Error Diagnostics on the Ordinary Held-Out Split",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "10_alcu_model_error_by_state.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    process_inputs = pd.read_csv(RAW_DIR / "X_pvd_AlCu.csv")
    targets = pd.read_csv(RAW_DIR / "Y_pvd_AlCu.csv")
    assignments = pd.read_csv(PROCESSED_DIR / "alcu_split_assignments.csv")
    monitoring = pd.read_csv(PROCESSED_DIR / "alcu_monitoring_scores.csv")
    feature_screening = pd.read_csv(PROCESSED_DIR / "alcu_feature_screening.csv")

    retained_features = feature_screening.loc[
        feature_screening["keep_for_modeling"], "feature"
    ].tolist()
    inputs_array = process_inputs[retained_features].to_numpy(dtype=float)
    targets_array = targets.to_numpy(dtype=float)
    valid_output = ~assignments["all_zero_output"].to_numpy()
    settings = [
        ("ordinary fixed split", "primary_split", False),
        (
            "identical-target-profile sensitivity",
            "grouped_sensitivity_split",
            True,
        ),
    ]
    metric_rows = []
    target_rows = []
    cv_tables = []
    forest_rows = []
    zero_rows = []
    primary_prediction_table = None

    for validation, split_column, grouped_cv in settings:
        split_train = assignments[split_column].eq("train").to_numpy()
        split_test = assignments[split_column].eq("test").to_numpy()
        train_mask = split_train & valid_output
        test_mask = split_test & valid_output
        reference_profile = np.median(targets_array[train_mask], axis=0)
        cv_groups = None
        if grouped_cv:
            cv_groups = assignments.loc[
                train_mask, "target_profile_group"
            ].to_numpy()
        selected_alpha, cv_results = choose_ridge_alpha(
            inputs_array[train_mask],
            targets_array[train_mask],
            cv_groups,
        )
        cv_results.insert(0, "validation", validation)
        cv_results["selected"] = cv_results["alpha"].eq(selected_alpha)
        cv_tables.append(cv_results)

        ridge_cv_rmse = float(
            cv_results.loc[cv_results["selected"], "mean_validation_rmse"].iat[0]
        )
        forest_gate = evaluate_random_forest_gate(
            inputs_array[train_mask],
            targets_array[train_mask],
            ridge_cv_rmse,
            cv_groups,
        )
        forest_gate.insert(0, "validation", validation)
        if not bool(forest_gate["passes_ten_percent_gate"].all()):
            raise ValueError("random forest did not pass its training-only entry gate")
        forest_rows.extend(forest_gate.to_dict("records"))

        models = [
            ("Linear Regression", np.nan, LinearRegression()),
            ("Ridge", selected_alpha, Ridge(alpha=selected_alpha)),
        ]
        for model_name, alpha, model in models:
            _, predictions = fit_and_assess(
                inputs_array,
                targets_array,
                train_mask,
                test_mask,
                model,
            )
            actual = targets_array[test_mask]
            summary = summarize_model(
                validation,
                "excluded",
                model_name,
                alpha,
                train_mask,
                test_mask,
                assignments,
                actual,
                predictions,
                reference_profile,
            )
            metric_rows.append(summary)
            target_rows.extend(
                summarize_targets(
                    validation,
                    model_name,
                    alpha,
                    targets.columns,
                    actual,
                    predictions,
                )
            )
        ridge_summary = next(
            row
            for row in metric_rows
            if row["validation"] == validation
            and row["zero_output_treatment"] == "excluded"
            and row["model"] == "Ridge"
        )
        selected_forest = make_random_forest(RANDOM_STATE)
        selected_forest.fit(inputs_array[train_mask], targets_array[train_mask])
        selected_forest_predictions = selected_forest.predict(
            inputs_array[test_mask]
        )
        forest_summary = summarize_model(
            validation,
            "excluded",
            "Random Forest",
            np.nan,
            train_mask,
            test_mask,
            assignments,
            actual,
            selected_forest_predictions,
            reference_profile,
        )
        metric_rows.append(forest_summary)
        target_rows.extend(
            summarize_targets(
                validation,
                "Random Forest",
                np.nan,
                targets.columns,
                actual,
                selected_forest_predictions,
            )
        )
        if validation == "ordinary fixed split":
            primary_actual = actual
            primary_predicted = selected_forest_predictions
            primary_prediction_table = build_prediction_table(
                assignments,
                monitoring,
                test_mask,
                targets.columns,
                actual,
                selected_forest_predictions,
                reference_profile,
            )

        # reuse model settings so this comparison isolates zero handling
        _, included_ridge_predictions = fit_and_assess(
            inputs_array,
            targets_array,
            split_train,
            split_test,
            Ridge(alpha=selected_alpha),
        )
        included_ridge_summary = summarize_model(
            validation,
            "included for sensitivity only",
            "Ridge",
            selected_alpha,
            split_train,
            split_test,
            assignments,
            targets_array[split_test],
            included_ridge_predictions,
            reference_profile,
        )
        metric_rows.append(included_ridge_summary)

        included_forest = make_random_forest(RANDOM_STATE)
        included_forest.fit(
            inputs_array[split_train],
            targets_array[split_train],
        )
        included_forest_predictions = included_forest.predict(
            inputs_array[split_test]
        )
        included_forest_summary = summarize_model(
            validation,
            "included for sensitivity only",
            "Random Forest",
            np.nan,
            split_train,
            split_test,
            assignments,
            targets_array[split_test],
            included_forest_predictions,
            reference_profile,
        )
        metric_rows.append(included_forest_summary)

        zero_output_location = (
            "train" if bool((split_train & ~valid_output).any()) else "test"
        )
        for model_name, alpha, excluded_summary, included_summary in [
            (
                "Ridge",
                selected_alpha,
                ridge_summary,
                included_ridge_summary,
            ),
            (
                "Random Forest",
                np.nan,
                forest_summary,
                included_forest_summary,
            ),
        ]:
            zero_rows.append(
                {
                    "validation": validation,
                    "model": model_name,
                    "zero_output_location": zero_output_location,
                    "ridge_alpha": alpha,
                    "excluded_training_rows": excluded_summary["training_rows"],
                    "included_training_rows": included_summary["training_rows"],
                    "excluded_assessment_rows": excluded_summary["assessment_rows"],
                    "included_assessment_rows": included_summary["assessment_rows"],
                    "excluded_aggregate_rmse": excluded_summary[
                        "aggregate_rmse_17_targets"
                    ],
                    "included_aggregate_rmse": included_summary[
                        "aggregate_rmse_17_targets"
                    ],
                    "rmse_change_included_minus_excluded": included_summary[
                        "aggregate_rmse_17_targets"
                    ]
                    - excluded_summary["aggregate_rmse_17_targets"],
                }
            )

    metrics = pd.DataFrame(metric_rows)
    per_target_metrics = pd.DataFrame(target_rows)
    ridge_cv_results = pd.concat(cv_tables, ignore_index=True)
    forest_stability = pd.DataFrame(forest_rows)
    zero_sensitivity = pd.DataFrame(zero_rows)
    if not bool(forest_stability["passes_ten_percent_gate"].all()):
        raise ValueError("random forest did not pass its predeclared entry gate")
    error_by_state = summarize_error_by_state(primary_prediction_table)
    metrics.to_csv(RESULTS_DIR / "alcu_virtual_metrology_metrics.csv", index=False)
    per_target_metrics.to_csv(
        RESULTS_DIR / "alcu_virtual_metrology_per_target.csv", index=False
    )
    ridge_cv_results.to_csv(
        RESULTS_DIR / "alcu_ridge_cv_results.csv", index=False
    )
    forest_stability.to_csv(
        RESULTS_DIR / "alcu_random_forest_stability.csv", index=False
    )
    zero_sensitivity.to_csv(
        RESULTS_DIR / "alcu_zero_output_model_sensitivity.csv", index=False
    )
    primary_prediction_table.to_csv(
        PROCESSED_DIR / "alcu_primary_model_predictions.csv", index=False
    )
    error_by_state.to_csv(
        RESULTS_DIR / "alcu_model_error_by_state.csv", index=False
    )

    primary_metrics = metrics.loc[
        metrics["validation"].eq("ordinary fixed split")
        & metrics["zero_output_treatment"].eq("excluded")
        & metrics["model"].eq("Random Forest")
    ].iloc[0]
    plot_primary_predictions(
        primary_actual,
        primary_predicted,
        primary_prediction_table,
        primary_metrics,
    )
    plot_per_target_errors(per_target_metrics)
    plot_error_by_state(primary_prediction_table)

    display_columns = [
        "validation",
        "zero_output_treatment",
        "model",
        "ridge_alpha",
        "training_rows",
        "assessment_rows",
        "aggregate_mae_17_targets",
        "aggregate_rmse_17_targets",
        "aggregate_r2_17_targets",
    ]
    print("AlCu direct 17-target virtual metrology complete.")
    print(metrics[display_columns].to_string(index=False))
    print()
    print("Ridge tuning:")
    print(
        ridge_cv_results.loc[
            ridge_cv_results["selected"],
            ["validation", "alpha", "mean_validation_rmse"],
        ].to_string(index=False)
    )
    print()
    print("Random Forest entry-gate stability:")
    print(
        forest_stability[
            [
                "validation",
                "random_state",
                "mean_validation_rmse",
                "ridge_mean_validation_rmse",
                "rmse_improvement_over_ridge",
                "passes_ten_percent_gate",
            ]
        ].to_string(index=False)
    )
    print()
    print("All-zero output sensitivity (no physical meaning assigned):")
    print(zero_sensitivity.to_string(index=False))
    print()
    print("Primary Random Forest error by retrospective process state:")
    print(error_by_state.to_string(index=False))


if __name__ == "__main__":
    main()
