from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


# paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ALCU_RESULTS_DIR = PROJECT_DIR / "results" / "alcu"
FIGURES_DIR = PROJECT_DIR / "figures"

# analysis settings
PROCESS = "AlCu"
EXPLAINED_VARIANCE_TARGET = 0.90
ALERT_QUANTILE = 0.99

# alarm categories and their plot colors
CATEGORY_ORDER = ["normal", "t2 alarm only", "q alarm only", "both alarms"]
CATEGORY_COLORS = {
    "normal": "#8A94A6",
    "t2 alarm only": "#C44E52",
    "q alarm only": "#E39C37",
    "both alarms": "#7A3E9D",
}


# monitoring statistics


# compute pca scores, t2, q and residuals for every row
def calculate_monitoring_statistics(standardized_data, components, eigenvalues):
    # scores describe each observation inside the retained pca process space
    scores = standardized_data @ components.T

    # t2 measures distance from the center along the retained variation directions
    t2_values = np.sum((scores**2) / eigenvalues, axis=1)

    # q measures standardized variation the retained pca model cannot reconstruct
    reconstructed = scores @ components
    residuals = standardized_data - reconstructed
    q_values = np.sum(residuals**2, axis=1)
    return scores, t2_values, q_values, residuals


# label each row as normal, t2 only, q only or both alarms
def assign_alarm_categories(t2_values, q_values, t2_limit, q_limit):
    t2_alarm = t2_values > t2_limit
    q_alarm = q_values > q_limit
    categories = np.full(len(t2_values), "normal", dtype=object)
    categories[t2_alarm & ~q_alarm] = "t2 alarm only"
    categories[~t2_alarm & q_alarm] = "q alarm only"
    categories[t2_alarm & q_alarm] = "both alarms"
    return categories


# figures


# save a figure into the figures folder and close it
def save_figure(figure, file_name):
    figure.savefig(
        FIGURES_DIR / file_name,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


# remove the top and right frame lines of an axis
def hide_top_right_spines(axis):
    axis.spines[["top", "right"]].set_visible(False)


# scatter two columns with one colored group per alarm category
def scatter_by_category(axis, monitoring_scores, x_column, y_column, marker_size):
    for category in CATEGORY_ORDER:
        group = monitoring_scores.loc[monitoring_scores["alarm_category"].eq(category)]
        if category == "normal":
            alpha = 0.60
        else:
            alpha = 0.85
        axis.scatter(
            group[x_column],
            group[y_column],
            s=marker_size,
            alpha=alpha,
            color=CATEGORY_COLORS[category],
            label=f"{category} (n={len(group)})",
        )


# bar and cumulative plots of the pca explained variance
def plot_explained_variance(explained_ratios, retained_components):
    component_numbers = np.arange(1, len(explained_ratios) + 1)
    cumulative_variance = np.cumsum(explained_ratios)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # variance of each component
    axes[0].bar(component_numbers, explained_ratios, color="#2F6690", width=0.85)
    axes[0].axvline(
        retained_components + 0.5,
        color="#C44E52",
        linestyle="--",
        linewidth=1.3,
        label=f"{retained_components} retained components",
    )
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("Variance Explained by Each Component")
    axes[0].legend(frameon=False)

    # running total against the retention target
    axes[1].plot(component_numbers, cumulative_variance, color="#2F6690", linewidth=2)
    axes[1].axhline(
        EXPLAINED_VARIANCE_TARGET,
        color="#C44E52",
        linestyle="--",
        linewidth=1.3,
        label="90% retention rule",
    )
    axes[1].axvline(retained_components, color="#C44E52", linestyle=":", linewidth=1.2)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Number of principal components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(axis="y", alpha=0.20)
        hide_top_right_spines(axis)

    figure.suptitle("AlCu PCA Process Representation", fontsize=14)
    figure.tight_layout()
    save_figure(figure, "02_alcu_pca_variance.png")


# first two pca scores colored by alarm category
def plot_pca_scores(monitoring_scores, explained_ratios):
    figure, axis = plt.subplots(figsize=(8.5, 6))
    scatter_by_category(axis, monitoring_scores, "pc1_score", "pc2_score", 16)

    axis.set_xlabel(f"PC1 score ({explained_ratios[0]:.1%} variance)")
    axis.set_ylabel(f"PC2 score ({explained_ratios[1]:.1%} variance)")
    axis.set_title("AlCu PCA Scores by Retrospective Process State")
    axis.grid(alpha=0.20)
    hide_top_right_spines(axis)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    save_figure(figure, "03_alcu_pca_scores.png")


# one statistic over the observation index, alerts highlighted
def plot_alert_panel(
    axis, monitoring_scores, alarm_column, ratio_column, alert_color, alert_label
):
    observation_index = monitoring_scores["observation_index"]
    alarm = monitoring_scores[alarm_column]
    axis.scatter(
        observation_index[~alarm],
        monitoring_scores.loc[~alarm, ratio_column],
        s=8,
        alpha=0.50,
        color="#8A94A6",
        label="below threshold",
    )
    axis.scatter(
        observation_index[alarm],
        monitoring_scores.loc[alarm, ratio_column],
        s=18,
        alpha=0.85,
        color=alert_color,
        label=alert_label,
    )
    axis.axhline(1, color=alert_color, linestyle="--", linewidth=1.3)
    axis.set_yscale("log")


# t2 and q ratios over the observation index
def plot_monitoring_statistics(monitoring_scores):
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # t2 panel
    plot_alert_panel(
        axes[0], monitoring_scores, "t2_alarm", "t2_ratio", "#C44E52", "T² alert"
    )
    axes[0].set_ylabel("T² / empirical threshold")
    axes[0].set_title("Retained-Space Distance")
    axes[0].legend(frameon=False)

    # q panel
    plot_alert_panel(
        axes[1], monitoring_scores, "q_alarm", "q_ratio", "#E39C37", "Q alert"
    )
    axes[1].set_xlabel("Observation index (not established as chronological)")
    axes[1].set_ylabel("Q / empirical threshold")
    axes[1].set_title("Residual-Space Distance")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.20)
        hide_top_right_spines(axis)

    figure.suptitle("AlCu Empirical Multivariate Alert Statistics", fontsize=14)
    figure.tight_layout()
    save_figure(figure, "04_alcu_monitoring_statistics.png")


# t2 ratio against q ratio colored by alarm category
def plot_process_state_map(monitoring_scores):
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    scatter_by_category(axis, monitoring_scores, "t2_ratio", "q_ratio", 17)

    axis.axvline(1, color="#C44E52", linestyle="--", linewidth=1.3)
    axis.axhline(1, color="#E39C37", linestyle="--", linewidth=1.3)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("T² / empirical T² threshold")
    axis.set_ylabel("Q / empirical Q threshold")
    axis.set_title("AlCu Retrospective Process-State Map")
    axis.grid(alpha=0.20, which="both")
    hide_top_right_spines(axis)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    save_figure(figure, "05_alcu_process_state_map.png")


# result tables


# one row per observation with scores, limits, ratios and alarms
def build_monitoring_scores(assignments, scores, t2_values, q_values, t2_limit, q_limit):
    alarm_categories = assign_alarm_categories(t2_values, q_values, t2_limit, q_limit)

    monitoring_scores = assignments.copy()
    monitoring_scores["pc1_score"] = scores[:, 0]
    monitoring_scores["pc2_score"] = scores[:, 1]
    monitoring_scores["t2"] = t2_values
    monitoring_scores["q"] = q_values
    monitoring_scores["t2_limit"] = t2_limit
    monitoring_scores["q_limit"] = q_limit
    monitoring_scores["t2_ratio"] = t2_values / t2_limit
    monitoring_scores["q_ratio"] = q_values / q_limit
    monitoring_scores["t2_alarm"] = t2_values > t2_limit
    monitoring_scores["q_alarm"] = q_values > q_limit
    monitoring_scores["alarm_category"] = alarm_categories
    return monitoring_scores


# explained and cumulative variance for every component
def build_component_table(explained_ratios, cumulative_variance, retained_component_count):
    component_numbers = np.arange(1, len(explained_ratios) + 1)
    return pd.DataFrame(
        {
            "principal_component": component_numbers,
            "explained_variance_ratio": explained_ratios,
            "cumulative_explained_variance": cumulative_variance,
            "retained": component_numbers <= retained_component_count,
        }
    )


# number and share of each alarm category within each split
def count_alarms_by_split(monitoring_scores):
    alarm_counts = (
        monitoring_scores.groupby(["primary_split", "alarm_category"], observed=False)
        .size()
        .rename("observations")
        .reset_index()
    )
    split_totals = alarm_counts.groupby("primary_split")["observations"].transform("sum")
    alarm_counts["share_within_split"] = alarm_counts["observations"] / split_totals
    return alarm_counts


# main


# fit the alcu pca monitor, save its tables and figures, then print a summary
def main():
    ALCU_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # load the raw inputs, the split and the screened features
    process_inputs = pd.read_csv(RAW_DIR / "X_pvd_AlCu.csv")
    assignments = pd.read_csv(PROCESSED_DIR / "alcu_split_assignments.csv")
    feature_screening = pd.read_csv(PROCESSED_DIR / "alcu_feature_screening.csv")

    retained_features = feature_screening.loc[
        feature_screening["keep_for_modeling"], "feature"
    ].tolist()
    input_matrix = process_inputs[retained_features]
    train_mask = assignments["primary_split"].eq("train").to_numpy()

    # fit scaling only on the reference rows to prevent assessment leakage
    scaler = StandardScaler()
    scaler.fit(input_matrix.loc[train_mask])
    standardized_all = scaler.transform(input_matrix)
    standardized_train = standardized_all[train_mask]

    # fit a full pca once, then retain the smallest set reaching 90% variance
    full_pca = PCA(svd_solver="full")
    full_pca.fit(standardized_train)
    explained_ratios = full_pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_ratios)
    retained_component_count = int(
        np.searchsorted(cumulative_variance, EXPLAINED_VARIANCE_TARGET) + 1
    )

    components = full_pca.components_[:retained_component_count]
    eigenvalues = full_pca.explained_variance_[:retained_component_count]
    scores, t2_values, q_values, residuals = calculate_monitoring_statistics(
        standardized_all,
        components,
        eigenvalues,
    )

    # empirical limits avoid unsupported normality and known-good assumptions
    t2_limit = float(np.quantile(t2_values[train_mask], ALERT_QUANTILE))
    q_limit = float(np.quantile(q_values[train_mask], ALERT_QUANTILE))

    # per-observation monitoring table
    monitoring_scores = build_monitoring_scores(
        assignments, scores, t2_values, q_values, t2_limit, q_limit
    )
    monitoring_scores.to_csv(
        PROCESSED_DIR / "alcu_monitoring_scores.csv",
        index=False,
    )

    # fitted model for the excursion analysis
    np.savez(
        PROCESSED_DIR / "alcu_pca_model.npz",
        feature_names=np.array(retained_features),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        components=components,
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained_ratios,
        t2_limit=np.array([t2_limit]),
        q_limit=np.array([q_limit]),
    )

    # explained variance table
    component_table = build_component_table(
        explained_ratios, cumulative_variance, retained_component_count
    )
    component_table.to_csv(
        ALCU_RESULTS_DIR / "alcu_pca_explained_variance.csv",
        index=False,
    )

    # alarm counts by split
    alarm_counts = count_alarms_by_split(monitoring_scores)
    alarm_counts.to_csv(
        ALCU_RESULTS_DIR / "alcu_alarm_category_counts.csv",
        index=False,
    )

    # one-row summary of the monitoring model
    assessment_categories = monitoring_scores.loc[~train_mask, "alarm_category"]
    summary = pd.DataFrame(
        [
            {
                "process": PROCESS,
                "input_features": process_inputs.shape[1],
                "model_features": len(retained_features),
                "reference_rows": int(train_mask.sum()),
                "assessment_rows": int((~train_mask).sum()),
                "retained_components": retained_component_count,
                "retained_explained_variance": float(
                    cumulative_variance[retained_component_count - 1]
                ),
                "t2_empirical_quantile": ALERT_QUANTILE,
                "t2_threshold": t2_limit,
                "q_empirical_quantile": ALERT_QUANTILE,
                "q_threshold": q_limit,
                "assessment_any_alarm_rate": float(
                    assessment_categories.ne("normal").mean()
                ),
            }
        ]
    )
    summary.to_csv(
        ALCU_RESULTS_DIR / "alcu_pca_monitoring_summary.csv",
        index=False,
    )

    # monitoring values of the all-zero output record
    zero_output_monitoring = monitoring_scores.loc[
        monitoring_scores["all_zero_output"],
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
    zero_output_monitoring.to_csv(
        ALCU_RESULTS_DIR / "alcu_zero_output_monitoring.csv",
        index=False,
    )

    # figures
    plot_explained_variance(explained_ratios, retained_component_count)
    plot_pca_scores(monitoring_scores, explained_ratios)
    plot_monitoring_statistics(monitoring_scores)
    plot_process_state_map(monitoring_scores)

    # printed report
    print("AlCu PCA monitoring complete.")
    print(summary.to_string(index=False))
    print()
    print("Alarm counts by split:")
    print(alarm_counts.to_string(index=False))
    print()
    print("All-zero output record retained for process-state inspection:")
    print(zero_output_monitoring.to_string(index=False))


if __name__ == "__main__":
    main()
