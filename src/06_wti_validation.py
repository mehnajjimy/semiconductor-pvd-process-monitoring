from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# folders for inputs, outputs and figures
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ALCU_RESULTS_DIR = PROJECT_DIR / "results" / "alcu"
WTI_RESULTS_DIR = PROJECT_DIR / "results" / "wti"
COMPARISON_RESULTS_DIR = PROJECT_DIR / "results" / "comparison"
FIGURES_DIR = PROJECT_DIR / "figures"

# monitoring and model settings (same as alcu, except the wider ridge grid)
EXPLAINED_VARIANCE_TARGET = 0.90
ALERT_QUANTILE = 0.99
RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 2000
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
    10_000_000_000.0,
    100_000_000_000.0,
    1_000_000_000_000.0,
]
FOREST_TREES = 300
FOREST_MINIMUM_LEAF_SIZE = 5
FOREST_MAX_FEATURE_SHARE = 0.70

# alarm categories and their plot colors
CATEGORY_ORDER = ["normal", "t2 alarm only", "q alarm only", "both alarms"]
CATEGORY_COLORS = {
    "normal": "#8A94A6",
    "t2 alarm only": "#C44E52",
    "q alarm only": "#E39C37",
    "both alarms": "#7A3E9D",
}

# the two validation settings: name, split column and whether cv is grouped
VALIDATION_SETTINGS = [
    ("ordinary fixed split", "primary_split", False),
    (
        "identical-target-profile sensitivity",
        "grouped_sensitivity_split",
        True,
    ),
]


# error measures and model builders


# mean absolute error over every value
def aggregate_mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


# root mean squared error over every value
def aggregate_rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# wrap a model so its inputs are standardized first
def make_scaled_model(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


# build the random forest used for every wti fit
def make_random_forest():
    # this setting is fixed from the primary alcu gate, then refit on wti
    return RandomForestRegressor(
        n_estimators=FOREST_TREES,
        min_samples_leaf=FOREST_MINIMUM_LEAF_SIZE,
        max_features=FOREST_MAX_FEATURE_SHARE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# stop unless the alcu stability gate passed with these exact forest settings
def verify_alcu_forest_configuration():
    gate_path = ALCU_RESULTS_DIR / "alcu_random_forest_stability.csv"
    gate = pd.read_csv(gate_path)
    primary = gate.loc[gate["validation"].eq("ordinary fixed split")].copy()

    # all three primary seeds must pass the gate
    passed = primary["passes_ten_percent_gate"].astype(str).str.lower().eq("true")
    if len(primary) != 3 or not bool(passed.all()):
        raise ValueError(
            "WTi requires the Random Forest to pass all three primary AlCu "
            "training-only gate checks"
        )

    # the selected seed must be present exactly once
    selected = primary.loc[primary["random_state"].eq(RANDOM_STATE)]
    if len(selected) != 1:
        raise ValueError("AlCu gate output does not contain the selected seed 42")

    # the gated settings must match the ones used here
    configuration = selected.iloc[0]
    configuration_matches = (
        int(configuration["trees"]) == FOREST_TREES
        and int(configuration["minimum_leaf_size"]) == FOREST_MINIMUM_LEAF_SIZE
        and np.isclose(
            float(configuration["maximum_feature_share"]),
            FOREST_MAX_FEATURE_SHARE,
        )
    )
    if not configuration_matches:
        raise ValueError("WTi Random Forest settings do not match the admitted AlCu model")


# pca process monitoring


# label each row by which of the t2 and q limits it exceeds
def assign_alarm_categories(t2_values, q_values, t2_limit, q_limit):
    t2_alarm = t2_values > t2_limit
    q_alarm = q_values > q_limit
    categories = np.full(len(t2_values), "normal", dtype=object)
    categories[t2_alarm & ~q_alarm] = "t2 alarm only"
    categories[~t2_alarm & q_alarm] = "q alarm only"
    categories[t2_alarm & q_alarm] = "both alarms"
    return categories


# fit pca monitoring on the training rows and score every row
def fit_monitoring_model(inputs, assignments, feature_names):
    # fit scaling on the training rows only
    train_mask = assignments["primary_split"].eq("train").to_numpy()
    scaler = StandardScaler()
    scaler.fit(inputs[train_mask])
    standardized = scaler.transform(inputs)

    # keep the smallest number of components reaching the variance target
    full_pca = PCA(svd_solver="full")
    full_pca.fit(standardized[train_mask])
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    component_count = int(
        np.searchsorted(cumulative_variance, EXPLAINED_VARIANCE_TARGET) + 1
    )
    components = full_pca.components_[:component_count]
    eigenvalues = full_pca.explained_variance_[:component_count]

    # t2 is the distance inside the pca space and q is what it cannot rebuild
    scores = standardized @ components.T
    t2_values = np.sum((scores**2) / eigenvalues, axis=1)
    residuals = standardized - scores @ components
    q_values = np.sum(residuals**2, axis=1)

    # empirical limits come from the training rows
    t2_limit = float(np.quantile(t2_values[train_mask], ALERT_QUANTILE))
    q_limit = float(np.quantile(q_values[train_mask], ALERT_QUANTILE))

    monitoring = build_monitoring_table(
        assignments,
        scores,
        t2_values,
        q_values,
        t2_limit,
        q_limit,
    )
    model = {
        "feature_names": np.array(feature_names),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": full_pca.explained_variance_ratio_,
        "t2_limit": np.array([t2_limit]),
        "q_limit": np.array([q_limit]),
    }
    return monitoring, model, cumulative_variance


# add scores, statistics, limits and alarm labels to the split table
def build_monitoring_table(assignments, scores, t2_values, q_values, t2_limit, q_limit):
    categories = assign_alarm_categories(
        t2_values,
        q_values,
        t2_limit,
        q_limit,
    )
    monitoring = assignments.copy()
    monitoring["pc1_score"] = scores[:, 0]
    monitoring["pc2_score"] = scores[:, 1]
    monitoring["t2"] = t2_values
    monitoring["q"] = q_values
    monitoring["t2_limit"] = t2_limit
    monitoring["q_limit"] = q_limit
    monitoring["t2_ratio"] = t2_values / t2_limit
    monitoring["q_ratio"] = q_values / q_limit
    monitoring["t2_alarm"] = t2_values > t2_limit
    monitoring["q_alarm"] = q_values > q_limit
    monitoring["alarm_category"] = categories
    return monitoring


# save the variance explained by each component
def write_explained_variance_table(pca_model, cumulative_variance):
    component_count = len(pca_model["eigenvalues"])
    explained_ratios = pca_model["explained_variance_ratio"]
    component_numbers = np.arange(1, len(explained_ratios) + 1)
    component_table = pd.DataFrame(
        {
            "principal_component": component_numbers,
            "explained_variance_ratio": explained_ratios,
            "cumulative_explained_variance": cumulative_variance,
            "retained": component_numbers <= component_count,
        }
    )
    component_table.to_csv(
        WTI_RESULTS_DIR / "wti_pca_explained_variance.csv",
        index=False,
    )


# save how many rows fall in each alarm category within each split
def write_alarm_category_counts(monitoring):
    alarm_counts = (
        monitoring.groupby(["primary_split", "alarm_category"])
        .size()
        .rename("observations")
        .reset_index()
    )
    split_totals = alarm_counts.groupby("primary_split")["observations"].transform("sum")
    alarm_counts["share_within_split"] = alarm_counts["observations"] / split_totals
    alarm_counts.to_csv(
        WTI_RESULTS_DIR / "wti_alarm_category_counts.csv",
        index=False,
    )


# save a one-row summary of the wti pca monitoring model
def write_pca_summary(
    monitoring,
    pca_model,
    cumulative_variance,
    input_features,
    model_features,
):
    component_count = len(pca_model["eigenvalues"])
    pca_train = monitoring["primary_split"].eq("train").to_numpy()
    pca_summary = pd.DataFrame(
        [
            {
                "process": "WTi",
                "input_features": input_features,
                "model_features": model_features,
                "reference_rows": int(pca_train.sum()),
                "assessment_rows": int((~pca_train).sum()),
                "retained_components": component_count,
                "retained_explained_variance": float(
                    cumulative_variance[component_count - 1]
                ),
                "t2_empirical_quantile": ALERT_QUANTILE,
                "t2_threshold": float(pca_model["t2_limit"][0]),
                "q_empirical_quantile": ALERT_QUANTILE,
                "q_threshold": float(pca_model["q_limit"][0]),
                "assessment_any_alarm_rate": float(
                    monitoring.loc[~pca_train, "alarm_category"].ne("normal").mean()
                ),
            }
        ]
    )
    pca_summary.to_csv(
        WTI_RESULTS_DIR / "wti_pca_monitoring_summary.csv",
        index=False,
    )
    return pca_summary


# save the monitoring results for the all-zero target rows
def write_zero_output_monitoring(monitoring):
    zero_monitoring = monitoring.loc[
        monitoring["all_zero_output"],
        [
            "observation_index",
            "primary_split",
            "t2",
            "q",
            "t2_ratio",
            "q_ratio",
            "alarm_category",
        ],
    ]
    zero_monitoring.to_csv(
        WTI_RESULTS_DIR / "wti_zero_output_monitoring.csv",
        index=False,
    )


# held-out quality comparison


# add up each target-profile group's alarm and normal values for the bootstrap
def summarize_quality_groups(assessment, metric):
    group_rows = []
    for group_number, group in assessment.groupby("target_profile_group"):
        alarm_values = group.loc[group["any_alarm"], metric].dropna()
        normal_values = group.loc[~group["any_alarm"], metric].dropna()
        group_rows.append(
            {
                "target_profile_group": group_number,
                "alarm_sum": float(alarm_values.sum()),
                "alarm_count": len(alarm_values),
                "normal_sum": float(normal_values.sum()),
                "normal_count": len(normal_values),
            }
        )
    return pd.DataFrame(group_rows)


# resample whole groups and collect the alarm minus normal mean difference
def bootstrap_group_differences(group_table):
    rng = np.random.default_rng(RANDOM_STATE)
    differences = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.integers(0, len(group_table), size=len(group_table))
        sample = group_table.iloc[sampled]
        alarm_count = sample["alarm_count"].sum()
        normal_count = sample["normal_count"].sum()

        # skip samples with no alarm rows or no normal rows
        if alarm_count > 0 and normal_count > 0:
            alarm_mean = sample["alarm_sum"].sum() / alarm_count
            normal_mean = sample["normal_sum"].sum() / normal_count
            differences.append(alarm_mean - normal_mean)
    return differences


# compare one quality metric between alarm and normal held-out rows
def clustered_quality_interval(assessment, metric):
    # resampling target-profile groups keeps exactly repeated outputs together
    group_table = summarize_quality_groups(assessment, metric)
    differences = bootstrap_group_differences(group_table)

    alarm_values = assessment.loc[assessment["any_alarm"], metric].dropna()
    normal_values = assessment.loc[~assessment["any_alarm"], metric].dropna()
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "metric": metric,
        "normal_observations": len(normal_values),
        "alarm_observations": len(alarm_values),
        "mean_difference_alarm_minus_normal": float(
            alarm_values.mean() - normal_values.mean()
        ),
        "group_bootstrap_95_percent_lower": float(lower),
        "group_bootstrap_95_percent_upper": float(upper),
        "median_difference_alarm_minus_normal": float(
            alarm_values.median() - normal_values.median()
        ),
        "bootstrap_samples_used": len(differences),
    }


# save the alarm versus normal quality comparison for valid held-out rows
def write_quality_comparison(monitoring, quality_metrics):
    quality_columns = [
        "observation_index",
        "released_mean_index",
        "reference_profile_shape_deviation",
    ]
    monitoring_with_quality = monitoring.merge(
        quality_metrics[quality_columns],
        on="observation_index",
        how="left",
        validate="one_to_one",
    )
    monitoring_with_quality["any_alarm"] = monitoring_with_quality[
        "alarm_category"
    ].ne("normal")

    # only valid held-out rows are compared
    quality_assessment = monitoring_with_quality.loc[
        monitoring_with_quality["primary_split"].eq("test")
        & ~monitoring_with_quality["all_zero_output"]
    ].copy()
    comparison_rows = []
    for metric in ["released_mean_index", "reference_profile_shape_deviation"]:
        comparison_rows.append(clustered_quality_interval(quality_assessment, metric))
    quality_comparison = pd.DataFrame(comparison_rows)
    quality_comparison.to_csv(
        WTI_RESULTS_DIR / "wti_held_out_quality_comparison.csv",
        index=False,
    )
    return quality_comparison


# virtual metrology models


# build the cross-validation folds, grouped when groups are given
def make_ridge_folds(inputs, targets, groups):
    if groups is None:
        splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(inputs))
        cv_method = "shuffled 5-fold"
    else:
        splitter = GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(inputs, targets, groups))
        cv_method = "grouped 5-fold by exact target profile"
    return folds, cv_method


# pick the ridge alpha with the lowest mean validation rmse
def choose_ridge_alpha(inputs, targets, groups=None):
    folds, cv_method = make_ridge_folds(inputs, targets, groups)

    # score every alpha on the same folds
    rows = []
    for alpha in RIDGE_ALPHAS:
        errors = []
        for train_rows, validation_rows in folds:
            model = make_scaled_model(Ridge(alpha=alpha))
            model.fit(inputs[train_rows], targets[train_rows])
            predicted = model.predict(inputs[validation_rows])
            errors.append(aggregate_rmse(targets[validation_rows], predicted))
        rows.append(
            {
                "alpha": alpha,
                "cross_validation_method": cv_method,
                "folds": len(errors),
                "mean_validation_rmse": float(np.mean(errors)),
                "standard_deviation_validation_rmse": float(
                    np.std(errors, ddof=1)
                ),
                "maximum_validation_rmse": float(np.max(errors)),
            }
        )

    # ties go to the smaller alpha
    results = pd.DataFrame(rows)
    selected = results.sort_values(["mean_validation_rmse", "alpha"]).iloc[0]
    return float(selected["alpha"]), results


# measure how far each row's shape is from the reference profile
def reference_profile_shape(values, reference_profile):
    values = np.asarray(values, dtype=float)
    point_relative = values / reference_profile
    row_relative_mean = point_relative.mean(axis=1)

    # rows with a zero or non-finite level have no shape and stay nan
    valid = np.isfinite(row_relative_mean) & ~np.isclose(row_relative_mean, 0)
    deviations = np.full(len(values), np.nan)
    shape_relative = point_relative[valid] / row_relative_mean[valid, None]
    deviations[valid] = np.sqrt(np.mean((shape_relative - 1) ** 2, axis=1))
    return deviations


# summarize one model's errors on one validation setting
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
    selection_source = {
        "Linear Regression": "required baseline",
        "Ridge": "training-only cross-validation",
        "Random Forest": "fixed after primary AlCu training-only stability gate",
    }[model_name]

    return {
        "validation": validation,
        "zero_output_treatment": zero_treatment,
        "model": model_name,
        "model_selection_source": selection_source,
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


# summarize one model's errors for each target separately
def summarize_targets(validation, model_name, alpha, names, actual, predicted):
    rows = []
    for target_number, target in enumerate(names):
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


# fit a scaled model on the training rows and predict the test rows
def fit_scaled_model(inputs, targets, train_mask, test_mask, model):
    fitted = make_scaled_model(model)
    fitted.fit(inputs[train_mask], targets[train_mask])
    return fitted.predict(inputs[test_mask])


# fit the random forest on the training rows and predict the test rows
def fit_forest(inputs, targets, train_mask, test_mask):
    fitted = make_random_forest()
    fitted.fit(inputs[train_mask], targets[train_mask])
    return fitted.predict(inputs[test_mask])


# build the per-row prediction table with each row's alarm state
def make_prediction_table(
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

    # row-level error and quality summaries
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

    # actual and predicted value for every target
    for target_number, target in enumerate(target_names):
        table[f"actual_{target}"] = actual[:, target_number]
        table[f"predicted_{target}"] = predicted[:, target_number]
    return table


# summarize row errors within each alarm category
def summarize_error_by_state(predictions):
    rows = []
    for category in CATEGORY_ORDER:
        values = predictions.loc[
            predictions["alarm_category"].eq(category),
            "row_mae_17_targets",
        ]

        mean_row_mae = np.nan
        median_row_mae = np.nan
        p90_row_mae = np.nan
        maximum_row_mae = np.nan
        if len(values) > 0:
            mean_row_mae = float(values.mean())
            median_row_mae = float(values.median())
            p90_row_mae = float(values.quantile(0.90))
            maximum_row_mae = float(values.max())

        rows.append(
            {
                "alarm_category": category,
                "observations": len(values),
                "mean_row_mae": mean_row_mae,
                "median_row_mae": median_row_mae,
                "p90_row_mae": p90_row_mae,
                "maximum_row_mae": maximum_row_mae,
            }
        )
    return pd.DataFrame(rows)


# fit the three models on the given rows and predict the test rows
def predict_with_all_models(
    inputs_array,
    targets_array,
    train_mask,
    test_mask,
    ridge_alpha,
):
    return {
        "Linear Regression": fit_scaled_model(
            inputs_array,
            targets_array,
            train_mask,
            test_mask,
            LinearRegression(),
        ),
        "Ridge": fit_scaled_model(
            inputs_array,
            targets_array,
            train_mask,
            test_mask,
            Ridge(alpha=ridge_alpha),
        ),
        "Random Forest": fit_forest(
            inputs_array,
            targets_array,
            train_mask,
            test_mask,
        ),
    }


# refit ridge and the forest with the all-zero record kept in
def predict_with_zero_rows_included(
    inputs_array,
    targets_array,
    split_train,
    split_test,
    ridge_alpha,
):
    return {
        "Ridge": fit_scaled_model(
            inputs_array,
            targets_array,
            split_train,
            split_test,
            Ridge(alpha=ridge_alpha),
        ),
        "Random Forest": fit_forest(
            inputs_array,
            targets_array,
            split_train,
            split_test,
        ),
    }


# compare one model's rmse with the all-zero record excluded and included
def compare_zero_treatment(
    validation,
    model_name,
    zero_location,
    alpha,
    excluded_summary,
    included_summary,
):
    return {
        "validation": validation,
        "model": model_name,
        "zero_output_location": zero_location,
        "ridge_alpha": alpha,
        "excluded_training_rows": excluded_summary["training_rows"],
        "included_training_rows": included_summary["training_rows"],
        "excluded_assessment_rows": excluded_summary["assessment_rows"],
        "included_assessment_rows": included_summary["assessment_rows"],
        "excluded_aggregate_rmse": excluded_summary["aggregate_rmse_17_targets"],
        "included_aggregate_rmse": included_summary["aggregate_rmse_17_targets"],
        "rmse_change_included_minus_excluded": included_summary[
            "aggregate_rmse_17_targets"
        ]
        - excluded_summary["aggregate_rmse_17_targets"],
    }


# run every model for one validation setting and collect the result rows
def run_validation_setting(
    validation,
    split_column,
    grouped_cv,
    inputs_array,
    targets_array,
    target_names,
    assignments,
    monitoring,
    valid_output,
):
    split_train = assignments[split_column].eq("train").to_numpy()
    split_test = assignments[split_column].eq("test").to_numpy()
    train_mask = split_train & valid_output
    test_mask = split_test & valid_output
    actual = targets_array[test_mask]
    reference_profile = np.median(targets_array[train_mask], axis=0)

    # choose the ridge alpha on the valid training rows only
    groups = None
    if grouped_cv:
        groups = assignments.loc[train_mask, "target_profile_group"].to_numpy()
    selected_alpha, cv_results = choose_ridge_alpha(
        inputs_array[train_mask],
        targets_array[train_mask],
        groups,
    )
    cv_results.insert(0, "validation", validation)
    cv_results["selected"] = cv_results["alpha"].eq(selected_alpha)

    # main comparison with the all-zero target record excluded
    metric_rows = []
    target_rows = []
    excluded_summaries = {}
    model_predictions = predict_with_all_models(
        inputs_array,
        targets_array,
        train_mask,
        test_mask,
        selected_alpha,
    )
    for model_name, predicted in model_predictions.items():
        alpha = selected_alpha if model_name == "Ridge" else np.nan
        excluded_summary = summarize_model(
            validation,
            "excluded",
            model_name,
            alpha,
            train_mask,
            test_mask,
            assignments,
            actual,
            predicted,
            reference_profile,
        )
        metric_rows.append(excluded_summary)
        excluded_summaries[model_name] = excluded_summary
        target_rows.extend(
            summarize_targets(
                validation,
                model_name,
                alpha,
                target_names,
                actual,
                predicted,
            )
        )

    # the ordinary split forest gives the primary prediction table
    primary = None
    if validation == "ordinary fixed split":
        primary_predicted = model_predictions["Random Forest"]
        primary_table = make_prediction_table(
            assignments,
            monitoring,
            test_mask,
            target_names,
            actual,
            primary_predicted,
            reference_profile,
        )
        primary = (actual, primary_predicted, primary_table)

    # sensitivity check with the all-zero target record included
    zero_rows = []
    included_predictions = predict_with_zero_rows_included(
        inputs_array,
        targets_array,
        split_train,
        split_test,
        selected_alpha,
    )
    zero_location = "train" if bool((split_train & ~valid_output).any()) else "test"
    for model_name, predicted in included_predictions.items():
        alpha = selected_alpha if model_name == "Ridge" else np.nan
        included_summary = summarize_model(
            validation,
            "included for sensitivity only",
            model_name,
            alpha,
            split_train,
            split_test,
            assignments,
            targets_array[split_test],
            predicted,
            reference_profile,
        )
        metric_rows.append(included_summary)
        zero_rows.append(
            compare_zero_treatment(
                validation,
                model_name,
                zero_location,
                alpha,
                excluded_summaries[model_name],
                included_summary,
            )
        )

    return metric_rows, target_rows, cv_results, zero_rows, primary


# cross-process comparison


# select the excluded-zero random forest rows for one validation setting
def select_forest_rows(metrics, validation):
    return metrics.loc[
        metrics["validation"].eq(validation)
        & metrics["zero_output_treatment"].eq("excluded")
        & metrics["model"].eq("Random Forest")
    ]


# build one process row for the cross-process comparison
def summarize_process(process, process_pca, process_vm):
    primary_rf = select_forest_rows(process_vm, "ordinary fixed split").iloc[0]

    # a missing grouped result is left as nan
    grouped_candidates = select_forest_rows(
        process_vm, "identical-target-profile sensitivity"
    )
    if grouped_candidates.empty:
        grouped_rmse = np.nan
        grouped_r2 = np.nan
    else:
        grouped_rf = grouped_candidates.iloc[0]
        grouped_rmse = float(grouped_rf["aggregate_rmse_17_targets"])
        grouped_r2 = float(grouped_rf["aggregate_r2_17_targets"])

    return {
        "process": process,
        "input_features": int(process_pca["input_features"]),
        "model_features": int(process_pca["model_features"]),
        "retained_components": int(process_pca["retained_components"]),
        "retained_explained_variance": float(
            process_pca["retained_explained_variance"]
        ),
        "primary_assessment_alarm_rate": float(
            process_pca["assessment_any_alarm_rate"]
        ),
        "primary_random_forest_mae": float(
            primary_rf["aggregate_mae_17_targets"]
        ),
        "primary_random_forest_rmse": float(
            primary_rf["aggregate_rmse_17_targets"]
        ),
        "primary_random_forest_r2": float(
            primary_rf["aggregate_r2_17_targets"]
        ),
        "grouped_random_forest_rmse": grouped_rmse,
        "grouped_random_forest_r2": grouped_r2,
        "grouped_minus_primary_rmse": float(
            grouped_rmse
            - primary_rf["aggregate_rmse_17_targets"]
        ),
    }


# put the alcu and wti results side by side and save them
def write_cross_process_comparison(pca_summary, metrics):
    alcu_pca = pd.read_csv(
        ALCU_RESULTS_DIR / "alcu_pca_monitoring_summary.csv"
    ).iloc[0]
    alcu_vm = pd.read_csv(
        ALCU_RESULTS_DIR / "alcu_virtual_metrology_metrics.csv"
    )
    comparison = pd.DataFrame(
        [
            summarize_process("AlCu", alcu_pca, alcu_vm),
            summarize_process("WTi", pca_summary.iloc[0], metrics),
        ]
    )
    comparison.to_csv(
        COMPARISON_RESULTS_DIR / "cross_process_workflow_comparison.csv",
        index=False,
    )
    return comparison


# figures


# plot t2 and q ratios for every row, colored by alarm category
def plot_process_state_map(monitoring):
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    for category in CATEGORY_ORDER:
        group = monitoring.loc[monitoring["alarm_category"].eq(category)]
        axis.scatter(
            group["t2_ratio"],
            group["q_ratio"],
            s=18,
            alpha=0.60 if category == "normal" else 0.88,
            color=CATEGORY_COLORS[category],
            label=f"{category} (n={len(group)})",
        )
    axis.axvline(1, color="#C44E52", linestyle="--", linewidth=1.3)
    axis.axhline(1, color="#E39C37", linestyle="--", linewidth=1.3)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("T² / empirical T² threshold")
    axis.set_ylabel("Q / empirical Q threshold")
    axis.set_title("WTi Retrospective Process-State Map")
    axis.grid(alpha=0.20, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "11_wti_process_state_map.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


# left panel: actual against predicted for all 17 targets
def plot_target_hexbin(figure, axis, actual, predicted):
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))
    cells = axis.hexbin(
        actual.ravel(),
        predicted.ravel(),
        gridsize=52,
        mincnt=1,
        cmap="Oranges",
    )
    axis.plot([lower, upper], [lower, upper], color="#2F6690", linestyle="--")
    axis.set_xlabel("Actual released target value")
    axis.set_ylabel("Predicted released target value")
    axis.set_title("All 17 Direct Targets")
    figure.colorbar(cells, ax=axis, label="Assessment cells per hexagon")


# right panel: actual against predicted mean index, colored by alarm category
def plot_mean_index_panel(axis, table):
    for category in CATEGORY_ORDER:
        group = table.loc[table["alarm_category"].eq(category)]
        axis.scatter(
            group["actual_released_mean_index"],
            group["predicted_released_mean_index"],
            s=20,
            alpha=0.55 if category == "normal" else 0.90,
            color=CATEGORY_COLORS[category],
            label=f"{category} (n={len(group)})",
        )
    mean_lower = float(
        min(
            table["actual_released_mean_index"].min(),
            table["predicted_released_mean_index"].min(),
        )
    )
    mean_upper = float(
        max(
            table["actual_released_mean_index"].max(),
            table["predicted_released_mean_index"].max(),
        )
    )
    axis.plot(
        [mean_lower, mean_upper],
        [mean_lower, mean_upper],
        color="#2F6690",
        linestyle="--",
    )
    axis.set_xlabel("Actual released-scale mean index")
    axis.set_ylabel("Predicted released-scale mean index")
    axis.set_title("Secondary Mean-Index Summary")
    axis.legend(frameon=False, fontsize=8)


# plot the primary random forest predictions on the held-out rows
def plot_predictions(actual, predicted, table, metrics):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    plot_target_hexbin(figure, axes[0], actual, predicted)
    plot_mean_index_panel(axes[1], table)
    for axis in axes:
        axis.grid(alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Separately Fitted WTi Random Forest on the Ordinary Held-Out Split\n"
        f"17-target RMSE = {metrics['aggregate_rmse_17_targets']:.4f}; "
        f"aggregate R² = {metrics['aggregate_r2_17_targets']:.3f}",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        f"Scope: {len(table)} valid held-out rows; all-zero target record excluded.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(
        FIGURES_DIR / "12_wti_virtual_metrology_predictions.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


# plot components, alert rates and r2 for alcu and wti side by side
def plot_cross_process_comparison(comparison):
    processes = comparison["process"].tolist()
    positions = np.arange(len(processes))
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    colors = ["#2F6690", "#C46D3B"]

    # retained pca components
    axes[0].bar(
        processes,
        comparison["retained_components"],
        color=colors,
    )
    axes[0].set_ylabel("Retained principal components")
    axes[0].set_title("90% PCA Representation")

    # held-out alert rates
    axes[1].bar(
        processes,
        100 * comparison["primary_assessment_alarm_rate"],
        color=colors,
    )
    axes[1].set_ylabel("Held-out observations alerted (%)")
    axes[1].set_title("Empirical Process-State Alerts")

    # random forest r2 on the ordinary and grouped splits
    width = 0.34
    axes[2].bar(
        positions - width / 2,
        comparison["primary_random_forest_r2"],
        width=width,
        color=colors,
        alpha=0.95,
        label="ordinary split",
    )
    axes[2].bar(
        positions + width / 2,
        comparison["grouped_random_forest_r2"],
        width=width,
        color=colors,
        alpha=0.48,
        label="grouped sensitivity",
    )
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(processes)
    axes[2].set_ylabel("Aggregate R² within each process")
    axes[2].set_title("Separately Fitted Predictive Signal")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "AlCu Primary Workflow and Separately Fitted WTi Replication",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Alert rates include all ordinary-split X assessment rows. R² excludes "
        "all-zero target rows; grouped bars use each grouped sensitivity split.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(
        FIGURES_DIR / "13_cross_process_workflow_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


# console summary


# print the main result tables
def print_summary(
    pca_summary,
    metrics,
    quality_comparison,
    zero_sensitivity,
    comparison,
):
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
    print("Separately fitted WTi replication workflow complete.")
    print(pca_summary.to_string(index=False))
    print()
    print(metrics[display_columns].to_string(index=False))
    print()
    print("WTi held-out any-alert minus normal comparisons:")
    print(quality_comparison.to_string(index=False))
    print()
    print("All-zero output sensitivity (no physical meaning assigned):")
    print(zero_sensitivity.to_string(index=False))
    print()
    print("Cross-process workflow comparison:")
    print(comparison.to_string(index=False))


# main workflow


# run the separately fitted wti replication of the alcu workflow
def main():
    WTI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    verify_alcu_forest_configuration()

    # load the wti data and the decisions from the preprocessing step
    process_inputs = pd.read_csv(RAW_DIR / "X_pvd_WTi.csv")
    targets = pd.read_csv(RAW_DIR / "Y_pvd_WTi.csv")
    assignments = pd.read_csv(PROCESSED_DIR / "wti_split_assignments.csv")
    quality_metrics = pd.read_csv(PROCESSED_DIR / "wti_quality_metrics.csv")
    feature_screening = pd.read_csv(PROCESSED_DIR / "wti_feature_screening.csv")
    retained_features = feature_screening.loc[
        feature_screening["keep_for_modeling"], "feature"
    ].tolist()
    inputs_array = process_inputs[retained_features].to_numpy(dtype=float)
    targets_array = targets.to_numpy(dtype=float)
    valid_output = ~assignments["all_zero_output"].to_numpy()

    # pca monitoring on the retained features
    monitoring, pca_model, cumulative_variance = fit_monitoring_model(
        inputs_array,
        assignments,
        retained_features,
    )
    monitoring.to_csv(PROCESSED_DIR / "wti_monitoring_scores.csv", index=False)
    np.savez(PROCESSED_DIR / "wti_pca_model.npz", **pca_model)
    write_explained_variance_table(pca_model, cumulative_variance)
    write_alarm_category_counts(monitoring)
    pca_summary = write_pca_summary(
        monitoring,
        pca_model,
        cumulative_variance,
        process_inputs.shape[1],
        len(retained_features),
    )
    write_zero_output_monitoring(monitoring)
    quality_comparison = write_quality_comparison(monitoring, quality_metrics)

    # virtual metrology on both validation settings
    metric_rows = []
    target_rows = []
    cv_tables = []
    zero_rows = []
    primary_actual = None
    primary_predicted = None
    primary_predictions = None
    for validation, split_column, grouped_cv in VALIDATION_SETTINGS:
        setting_metrics, setting_targets, cv_results, setting_zero_rows, primary = (
            run_validation_setting(
                validation,
                split_column,
                grouped_cv,
                inputs_array,
                targets_array,
                targets.columns,
                assignments,
                monitoring,
                valid_output,
            )
        )
        metric_rows.extend(setting_metrics)
        target_rows.extend(setting_targets)
        cv_tables.append(cv_results)
        zero_rows.extend(setting_zero_rows)
        if primary is not None:
            primary_actual, primary_predicted, primary_predictions = primary

    # save the model results
    metrics = pd.DataFrame(metric_rows)
    per_target = pd.DataFrame(target_rows)
    ridge_cv = pd.concat(cv_tables, ignore_index=True)
    zero_sensitivity = pd.DataFrame(zero_rows)
    error_by_state = summarize_error_by_state(primary_predictions)
    metrics.to_csv(
        WTI_RESULTS_DIR / "wti_virtual_metrology_metrics.csv",
        index=False,
    )
    per_target.to_csv(
        WTI_RESULTS_DIR / "wti_virtual_metrology_per_target.csv",
        index=False,
    )
    ridge_cv.to_csv(
        WTI_RESULTS_DIR / "wti_ridge_cv_results.csv",
        index=False,
    )
    zero_sensitivity.to_csv(
        WTI_RESULTS_DIR / "wti_zero_output_model_sensitivity.csv",
        index=False,
    )
    error_by_state.to_csv(
        WTI_RESULTS_DIR / "wti_model_error_by_state.csv",
        index=False,
    )
    primary_predictions.to_csv(
        PROCESSED_DIR / "wti_primary_model_predictions.csv",
        index=False,
    )

    comparison = write_cross_process_comparison(pca_summary, metrics)

    # figures
    primary_metrics = select_forest_rows(metrics, "ordinary fixed split").iloc[0]
    plot_process_state_map(monitoring)
    plot_predictions(
        primary_actual,
        primary_predicted,
        primary_predictions,
        primary_metrics,
    )
    plot_cross_process_comparison(comparison)

    print_summary(
        pca_summary,
        metrics,
        quality_comparison,
        zero_sensitivity,
        comparison,
    )


if __name__ == "__main__":
    main()
