from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ALCU_RESULTS_DIR = PROJECT_DIR / "results" / "alcu"
FIGURES_DIR = PROJECT_DIR / "figures"

RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 2000
TOP_FEATURES_TO_PLOT = 12
CATEGORY_ORDER = ["normal", "t2 alarm only", "q alarm only", "both alarms"]
CATEGORY_COLORS = {
    "normal": "#8A94A6",
    "t2 alarm only": "#C44E52",
    "q alarm only": "#E39C37",
    "both alarms": "#7A3E9D",
}


def choose_excursions(monitoring):
    # one case from each alert type keeps the diagnostic set focused
    t2_case = monitoring.loc[monitoring["alarm_category"].eq("t2 alarm only")].nlargest(
        1, "t2_ratio"
    )
    q_case = monitoring.loc[monitoring["alarm_category"].eq("q alarm only")].nlargest(
        1, "q_ratio"
    )
    both_candidates = monitoring.loc[monitoring["alarm_category"].eq("both alarms")].copy()
    both_candidates["joint_alert_score"] = np.sqrt(
        both_candidates["t2_ratio"] * both_candidates["q_ratio"]
    )
    both_case = both_candidates.nlargest(1, "joint_alert_score")
    return pd.concat([t2_case, q_case, both_case], ignore_index=True)


def calculate_contributions(process_inputs, selected_excursions, model):
    feature_names = model["feature_names"].tolist()
    scaler_mean = model["scaler_mean"]
    scaler_scale = model["scaler_scale"]
    components = model["components"]
    eigenvalues = model["eigenvalues"]
    contribution_rows = []

    for _, excursion in selected_excursions.iterrows():
        observation_index = int(excursion["observation_index"])
        standardized_row = (
            process_inputs.loc[observation_index, feature_names].to_numpy(dtype=float)
            - scaler_mean
        ) / scaler_scale

        scores = components @ standardized_row
        reconstruction = scores @ components
        residual = standardized_row - reconstruction

        # squared residual contributions add exactly to q for this observation
        q_contributions = residual**2

        # signed t2 terms add to t2 but are diagnostic, not unique causal effects
        t2_direction = components.T @ (scores / eigenvalues)
        t2_contributions = standardized_row * t2_direction

        for statistic, contributions in [
            ("T2", t2_contributions),
            ("Q", q_contributions),
        ]:
            ranks = pd.Series(np.abs(contributions)).rank(
                method="first", ascending=False
            )
            for feature_number, feature in enumerate(feature_names):
                contribution_rows.append(
                    {
                        "observation_index": observation_index,
                        "alarm_category": excursion["alarm_category"],
                        "primary_split": excursion["primary_split"],
                        "statistic": statistic,
                        "feature": feature,
                        "contribution": float(contributions[feature_number]),
                        "absolute_contribution": float(abs(contributions[feature_number])),
                        "absolute_rank": int(ranks.iloc[feature_number]),
                    }
                )

    return pd.DataFrame(contribution_rows)


def plot_contribution_panel(axis, contribution_table, observation_index, statistic, title):
    selected = contribution_table.loc[
        contribution_table["observation_index"].eq(observation_index)
        & contribution_table["statistic"].eq(statistic)
    ].nsmallest(TOP_FEATURES_TO_PLOT, "absolute_rank")
    selected = selected.sort_values("contribution")

    if statistic == "T2":
        colors = np.where(selected["contribution"] >= 0, "#C44E52", "#2F6690")
    else:
        colors = "#E39C37"

    axis.barh(selected["feature"], selected["contribution"], color=colors)
    axis.axvline(0, color="#444444", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel(f"{statistic} contribution")
    axis.grid(axis="x", alpha=0.20)
    axis.spines[["top", "right"]].set_visible(False)


def plot_excursion_contributions(contribution_table, selected_excursions):
    def diagnostic_scope(observation_index):
        split = selected_excursions.loc[
            selected_excursions["observation_index"].eq(observation_index),
            "primary_split",
        ].iat[0]
        return "training diagnostic" if split == "train" else "held-out diagnostic"

    t2_index = int(
        selected_excursions.loc[
            selected_excursions["alarm_category"].eq("t2 alarm only"),
            "observation_index",
        ].iat[0]
    )
    q_index = int(
        selected_excursions.loc[
            selected_excursions["alarm_category"].eq("q alarm only"),
            "observation_index",
        ].iat[0]
    )
    both_index = int(
        selected_excursions.loc[
            selected_excursions["alarm_category"].eq("both alarms"),
            "observation_index",
        ].iat[0]
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_contribution_panel(
        axes[0, 0],
        contribution_table,
        t2_index,
        "T2",
        f"T²-Only Excursion {t2_index} ({diagnostic_scope(t2_index)})\n"
        "Signed T² Terms",
    )
    plot_contribution_panel(
        axes[0, 1],
        contribution_table,
        q_index,
        "Q",
        f"Q-Only Excursion {q_index} ({diagnostic_scope(q_index)})\n"
        "Squared Residual Terms",
    )
    plot_contribution_panel(
        axes[1, 0],
        contribution_table,
        both_index,
        "T2",
        f"Both-Alarm Excursion {both_index} ({diagnostic_scope(both_index)})\n"
        "Signed T² Terms",
    )
    plot_contribution_panel(
        axes[1, 1],
        contribution_table,
        both_index,
        "Q",
        f"Both-Alarm Excursion {both_index} ({diagnostic_scope(both_index)})\n"
        "Squared Residual Terms",
    )
    figure.suptitle(
        "AlCu Measurement Channels Prioritized for Engineering Review",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "06_alcu_excursion_contributions.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def combine_quality_and_monitoring(quality_metrics, monitoring):
    metric_columns = [
        "observation_index",
        "released_mean_index",
        "released_standard_deviation",
        "released_range",
        "released_coefficient_of_variation",
        "reference_profile_shape_deviation",
    ]
    combined = monitoring.merge(
        quality_metrics[metric_columns],
        on="observation_index",
        how="left",
        validate="one_to_one",
    )
    combined["any_alarm"] = combined["alarm_category"].ne("normal")
    return combined


def summarize_quality_by_category(combined):
    rows = []
    valid = combined.loc[~combined["all_zero_output"]].copy()
    scopes = {
        "all valid observations": valid,
        "primary assessment only": valid.loc[valid["primary_split"].eq("test")],
    }
    metrics = ["released_mean_index", "reference_profile_shape_deviation"]

    for scope_name, scope_data in scopes.items():
        for category in CATEGORY_ORDER:
            category_data = scope_data.loc[scope_data["alarm_category"].eq(category)]
            for metric in metrics:
                values = category_data[metric].dropna()
                rows.append(
                    {
                        "scope": scope_name,
                        "alarm_category": category,
                        "metric": metric,
                        "observations": len(values),
                        "mean": float(values.mean()) if len(values) else np.nan,
                        "standard_deviation": float(values.std(ddof=1))
                        if len(values) > 1
                        else np.nan,
                        "median": float(values.median()) if len(values) else np.nan,
                        "minimum": float(values.min()) if len(values) else np.nan,
                        "maximum": float(values.max()) if len(values) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def clustered_mean_difference_interval(assessment, metric):
    # resampling exact target-profile groups keeps repeated outputs clustered
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

    group_table = pd.DataFrame(group_rows)
    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap_differences = []
    group_count = len(group_table)

    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_rows = rng.integers(0, group_count, size=group_count)
        sample = group_table.iloc[sampled_rows]
        alarm_count = sample["alarm_count"].sum()
        normal_count = sample["normal_count"].sum()
        if alarm_count > 0 and normal_count > 0:
            alarm_mean = sample["alarm_sum"].sum() / alarm_count
            normal_mean = sample["normal_sum"].sum() / normal_count
            bootstrap_differences.append(alarm_mean - normal_mean)

    alarm_values = assessment.loc[assessment["any_alarm"], metric].dropna()
    normal_values = assessment.loc[~assessment["any_alarm"], metric].dropna()
    observed_difference = float(alarm_values.mean() - normal_values.mean())
    median_difference = float(alarm_values.median() - normal_values.median())
    lower, upper = np.quantile(bootstrap_differences, [0.025, 0.975])

    return {
        "metric": metric,
        "normal_observations": len(normal_values),
        "alarm_observations": len(alarm_values),
        "mean_difference_alarm_minus_normal": observed_difference,
        "group_bootstrap_95_percent_lower": float(lower),
        "group_bootstrap_95_percent_upper": float(upper),
        "median_difference_alarm_minus_normal": median_difference,
        "bootstrap_samples_used": len(bootstrap_differences),
    }


def build_zero_record_sensitivity(combined):
    rows = []
    for include_zero in [False, True]:
        if include_zero:
            data = combined.copy()
            treatment = "included where metric is defined"
        else:
            data = combined.loc[~combined["all_zero_output"]].copy()
            treatment = "excluded"

        for metric in ["released_mean_index", "reference_profile_shape_deviation"]:
            alarm_values = data.loc[data["any_alarm"], metric].dropna()
            normal_values = data.loc[~data["any_alarm"], metric].dropna()
            rows.append(
                {
                    "zero_output_treatment": treatment,
                    "metric": metric,
                    "normal_observations": len(normal_values),
                    "alarm_observations": len(alarm_values),
                    "normal_mean": float(normal_values.mean()),
                    "alarm_mean": float(alarm_values.mean()),
                    "mean_difference_alarm_minus_normal": float(
                        alarm_values.mean() - normal_values.mean()
                    ),
                    "zero_output_has_defined_metric": bool(
                        combined.loc[combined["all_zero_output"], metric].notna().all()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_quality_by_state(combined):
    valid = combined.loc[~combined["all_zero_output"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [
        ("released_mean_index", "Released-scale mean index"),
        ("reference_profile_shape_deviation", "Reference-profile shape deviation"),
    ]

    for axis, (metric, label) in zip(axes, metrics):
        values = [
            valid.loc[valid["alarm_category"].eq(category), metric].dropna().to_numpy()
            for category in CATEGORY_ORDER
        ]
        boxplot = axis.boxplot(values, tick_labels=CATEGORY_ORDER, patch_artist=True)
        for box, category in zip(boxplot["boxes"], CATEGORY_ORDER):
            box.set_facecolor(CATEGORY_COLORS[category])
            box.set_alpha(0.65)
        axis.set_ylabel(label)
        axis.set_xlabel("Retrospective process state")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "AlCu Released-Scale Output Summaries by Process State\n"
        "All valid training and held-out observations; all-zero target record "
        "excluded",
        fontsize=13,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "07_alcu_quality_by_state.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    ALCU_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    process_inputs = pd.read_csv(RAW_DIR / "X_pvd_AlCu.csv")
    monitoring = pd.read_csv(PROCESSED_DIR / "alcu_monitoring_scores.csv")
    quality_metrics = pd.read_csv(PROCESSED_DIR / "alcu_quality_metrics.csv")
    model = np.load(PROCESSED_DIR / "alcu_pca_model.npz", allow_pickle=False)

    selected_excursions = choose_excursions(monitoring)
    contributions = calculate_contributions(process_inputs, selected_excursions, model)
    combined = combine_quality_and_monitoring(quality_metrics, monitoring)

    quality_summary = summarize_quality_by_category(combined)
    assessment = combined.loc[
        combined["primary_split"].eq("test") & ~combined["all_zero_output"]
    ].copy()
    held_out_comparisons = pd.DataFrame(
        [
            clustered_mean_difference_interval(assessment, metric)
            for metric in [
                "released_mean_index",
                "reference_profile_shape_deviation",
            ]
        ]
    )
    zero_sensitivity = build_zero_record_sensitivity(combined)

    selected_columns = [
        "observation_index",
        "primary_split",
        "all_zero_output",
        "alarm_category",
        "t2",
        "q",
        "t2_ratio",
        "q_ratio",
    ]
    selected_excursions[selected_columns].to_csv(
        ALCU_RESULTS_DIR / "alcu_selected_excursions.csv",
        index=False,
    )
    contributions.to_csv(
        ALCU_RESULTS_DIR / "alcu_excursion_contributions.csv",
        index=False,
    )
    quality_summary.to_csv(
        ALCU_RESULTS_DIR / "alcu_quality_by_alarm_category.csv",
        index=False,
    )
    held_out_comparisons.to_csv(
        ALCU_RESULTS_DIR / "alcu_held_out_quality_comparison.csv",
        index=False,
    )
    zero_sensitivity.to_csv(
        ALCU_RESULTS_DIR / "alcu_zero_record_quality_sensitivity.csv",
        index=False,
    )

    plot_excursion_contributions(contributions, selected_excursions)
    plot_quality_by_state(combined)

    print("AlCu excursion and released-output analysis complete.")
    print("Selected excursions:")
    print(selected_excursions[selected_columns].to_string(index=False))
    print()
    print("Held-out any-alert minus normal comparisons:")
    print(held_out_comparisons.to_string(index=False))
    print()
    print("All-zero output sensitivity:")
    print(zero_sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
