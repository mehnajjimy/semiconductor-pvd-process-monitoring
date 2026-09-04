from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

PROCESS = "AlCu"
EXPLAINED_VARIANCE_TARGET = 0.90
ALERT_QUANTILE = 0.99

CATEGORY_ORDER = ["normal", "t2 alarm only", "q alarm only", "both alarms"]
CATEGORY_COLORS = {
    "normal": "#8A94A6",
    "t2 alarm only": "#C44E52",
    "q alarm only": "#E39C37",
    "both alarms": "#7A3E9D",
}


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


def assign_alarm_categories(t2_values, q_values, t2_limit, q_limit):
    t2_alarm = t2_values > t2_limit
    q_alarm = q_values > q_limit
    categories = np.full(len(t2_values), "normal", dtype=object)
    categories[t2_alarm & ~q_alarm] = "t2 alarm only"
    categories[~t2_alarm & q_alarm] = "q alarm only"
    categories[t2_alarm & q_alarm] = "both alarms"
    return categories


def plot_explained_variance(explained_ratios, retained_components):
    component_numbers = np.arange(1, len(explained_ratios) + 1)
    cumulative_variance = np.cumsum(explained_ratios)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

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
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("AlCu PCA Process Representation", fontsize=14)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "02_alcu_pca_variance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_pca_scores(monitoring_scores, explained_ratios):
    figure, axis = plt.subplots(figsize=(8.5, 6))

    for category in CATEGORY_ORDER:
        group = monitoring_scores.loc[monitoring_scores["alarm_category"].eq(category)]
        axis.scatter(
            group["pc1_score"],
            group["pc2_score"],
            s=16,
            alpha=0.60 if category == "normal" else 0.85,
            color=CATEGORY_COLORS[category],
            label=f"{category} (n={len(group)})",
        )

    axis.set_xlabel(f"PC1 score ({explained_ratios[0]:.1%} variance)")
    axis.set_ylabel(f"PC2 score ({explained_ratios[1]:.1%} variance)")
    axis.set_title("AlCu PCA Scores by Retrospective Process State")
    axis.grid(alpha=0.20)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "03_alcu_pca_scores.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_monitoring_statistics(monitoring_scores):
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    observation_index = monitoring_scores["observation_index"]

    t2_alarm = monitoring_scores["t2_alarm"]
    axes[0].scatter(
        observation_index[~t2_alarm],
        monitoring_scores.loc[~t2_alarm, "t2_ratio"],
        s=8,
        alpha=0.50,
        color="#8A94A6",
        label="below threshold",
    )
    axes[0].scatter(
        observation_index[t2_alarm],
        monitoring_scores.loc[t2_alarm, "t2_ratio"],
        s=18,
        alpha=0.85,
        color="#C44E52",
        label="T² alert",
    )
    axes[0].axhline(1, color="#C44E52", linestyle="--", linewidth=1.3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("T² / empirical threshold")
    axes[0].set_title("Retained-Space Distance")
    axes[0].legend(frameon=False)

    q_alarm = monitoring_scores["q_alarm"]
    axes[1].scatter(
        observation_index[~q_alarm],
        monitoring_scores.loc[~q_alarm, "q_ratio"],
        s=8,
        alpha=0.50,
        color="#8A94A6",
        label="below threshold",
    )
    axes[1].scatter(
        observation_index[q_alarm],
        monitoring_scores.loc[q_alarm, "q_ratio"],
        s=18,
        alpha=0.85,
        color="#E39C37",
        label="Q alert",
    )
    axes[1].axhline(1, color="#E39C37", linestyle="--", linewidth=1.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Observation index (not established as chronological)")
    axes[1].set_ylabel("Q / empirical threshold")
    axes[1].set_title("Residual-Space Distance")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("AlCu Empirical Multivariate Alert Statistics", fontsize=14)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "04_alcu_monitoring_statistics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_process_state_map(monitoring_scores):
    figure, axis = plt.subplots(figsize=(8.5, 6.5))

    for category in CATEGORY_ORDER:
        group = monitoring_scores.loc[monitoring_scores["alarm_category"].eq(category)]
        axis.scatter(
            group["t2_ratio"],
            group["q_ratio"],
            s=17,
            alpha=0.60 if category == "normal" else 0.85,
            color=CATEGORY_COLORS[category],
            label=f"{category} (n={len(group)})",
        )

    axis.axvline(1, color="#C44E52", linestyle="--", linewidth=1.3)
    axis.axhline(1, color="#E39C37", linestyle="--", linewidth=1.3)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("T² / empirical T² threshold")
    axis.set_ylabel("Q / empirical Q threshold")
    axis.set_title("AlCu Retrospective Process-State Map")
    axis.grid(alpha=0.20, which="both")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "05_alcu_process_state_map.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
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
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
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
    monitoring_scores.to_csv(
        PROCESSED_DIR / "alcu_monitoring_scores.csv",
        index=False,
    )

    np.savez(
        PROCESSED_DIR / "alcu_pca_model.npz",
        feature_names=np.array(retained_features),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        components=components,
        eigenvalues=eigenvalues,
        explained_variance_ratio=full_pca.explained_variance_ratio_,
        t2_limit=np.array([t2_limit]),
        q_limit=np.array([q_limit]),
    )

    component_table = pd.DataFrame(
        {
            "principal_component": np.arange(1, len(full_pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": full_pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative_variance,
            "retained": np.arange(1, len(full_pca.explained_variance_ratio_) + 1)
            <= retained_component_count,
        }
    )
    component_table.to_csv(RESULTS_DIR / "alcu_pca_explained_variance.csv", index=False)

    alarm_counts = (
        monitoring_scores.groupby(["primary_split", "alarm_category"], observed=False)
        .size()
        .rename("observations")
        .reset_index()
    )
    alarm_counts["share_within_split"] = alarm_counts["observations"] / alarm_counts.groupby(
        "primary_split"
    )["observations"].transform("sum")
    alarm_counts.to_csv(RESULTS_DIR / "alcu_alarm_category_counts.csv", index=False)

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
                    monitoring_scores.loc[~train_mask, "alarm_category"].ne("normal").mean()
                ),
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "alcu_pca_monitoring_summary.csv", index=False)

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
        RESULTS_DIR / "alcu_zero_output_monitoring.csv",
        index=False,
    )

    plot_explained_variance(
        full_pca.explained_variance_ratio_,
        retained_component_count,
    )
    plot_pca_scores(monitoring_scores, full_pca.explained_variance_ratio_)
    plot_monitoring_statistics(monitoring_scores)
    plot_process_state_map(monitoring_scores)

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
