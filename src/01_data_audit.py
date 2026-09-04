from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
AUDIT_RESULTS_DIR = PROJECT_DIR / "results" / "audit"

FILES = {
    "X_AlCu": RAW_DIR / "X_pvd_AlCu.csv",
    "Y_AlCu": RAW_DIR / "Y_pvd_AlCu.csv",
    "X_WTi": RAW_DIR / "X_pvd_WTi.csv",
    "Y_WTi": RAW_DIR / "Y_pvd_WTi.csv",
}

EXPECTED_MD5 = {
    "X_AlCu": "9dd4bfc44ffb6a570e519000e07e9fe8",
    "Y_AlCu": "99b845acd32ae57060e194e407c08f19",
    "X_WTi": "cfc27ea3bdf780bcdf6dfdc5711836c1",
    "Y_WTi": "ab6a3b425d18d6f88f0466a7c40b340a",
}


def calculate_md5(path):
    # read the file in chunks so the checksum works without loading the csv twice
    file_hash = md5()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def find_duplicate_columns(data):
    # compare full columns because equal correlations do not prove exact duplication
    duplicates = {}
    columns = list(data.columns)
    for column_number, column in enumerate(columns):
        for earlier_column in columns[:column_number]:
            if data[column].equals(data[earlier_column]):
                duplicates[column] = earlier_column
                break
    return duplicates


def longest_identical_run(data):
    # consecutive repeated targets can reveal grouping that a random split may separate
    longest_start = 0
    longest_end = 0
    current_start = 0

    for row_number in range(1, len(data)):
        if not data.iloc[row_number].equals(data.iloc[row_number - 1]):
            if row_number - current_start > longest_end - longest_start + 1:
                longest_start = current_start
                longest_end = row_number - 1
            current_start = row_number

    if len(data) - current_start > longest_end - longest_start + 1:
        longest_start = current_start
        longest_end = len(data) - 1

    return longest_start, longest_end, longest_end - longest_start + 1


def audit_file(file_label, path):
    # keep the audit read-only and calculate every check from the published file
    data = pd.read_csv(path)
    numeric_data = data.select_dtypes(include="number")
    values = numeric_data.to_numpy(dtype=float)
    duplicate_columns = find_duplicate_columns(data)
    observed_md5 = calculate_md5(path)

    summary = {
        "file": path.name,
        "dataset": file_label,
        "rows": len(data),
        "columns": data.shape[1],
        "numeric_columns": numeric_data.shape[1],
        "missing_cells": int(data.isna().sum().sum()),
        "nonfinite_cells": int((~np.isfinite(values)).sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "constant_columns": int((data.nunique(dropna=False) <= 1).sum()),
        "duplicate_columns": len(duplicate_columns),
        "minimum": float(np.nanmin(values)),
        "maximum": float(np.nanmax(values)),
        "outside_zero_one": int(((values < 0) | (values >= 1)).sum()),
        "observed_md5": observed_md5,
        "official_md5": EXPECTED_MD5[file_label],
        "official_checksum_match": observed_md5 == EXPECTED_MD5[file_label],
    }
    return data, summary, duplicate_columns


def build_feature_quality(process, data, duplicate_columns):
    # modal share highlights rare-state channels that standardization can overemphasize
    rows = []
    for column in data.columns:
        value_counts = data[column].value_counts(dropna=False)
        modal_count = int(value_counts.iloc[0])
        modal_value = value_counts.index[0]
        rows.append(
            {
                "process": process,
                "feature": column,
                "unique_values": int(data[column].nunique(dropna=False)),
                "modal_value": float(modal_value),
                "modal_count": modal_count,
                "modal_share": modal_count / len(data),
                "zero_share": float((data[column] == 0).mean()),
                "mean": float(data[column].mean()),
                "standard_deviation": float(data[column].std(ddof=0)),
                "minimum": float(data[column].min()),
                "maximum": float(data[column].max()),
                "constant": bool(data[column].nunique(dropna=False) <= 1),
                "near_constant_98_percent": bool(modal_count / len(data) >= 0.98),
                "dominant_value_80_percent": bool(modal_count / len(data) >= 0.80),
                "duplicate_of": duplicate_columns.get(column, ""),
            }
        )
    return rows


def audit_process(process, x_data, y_data):
    # x and y have no stored key, so only positional pairing can be checked
    combined = pd.concat([x_data, y_data], axis=1)
    zero_target_mask = (y_data == 0).all(axis=1)
    target_group_sizes = y_data.groupby(
        list(y_data.columns),
        dropna=False,
        sort=False,
    ).size()
    run_start, run_end, run_length = longest_identical_run(y_data)

    return {
        "process": process,
        "x_rows": len(x_data),
        "y_rows": len(y_data),
        "row_counts_match": len(x_data) == len(y_data),
        "stored_join_key_available": False,
        "positional_pairing_required": True,
        "combined_duplicate_rows": int(combined.duplicated().sum()),
        "unique_target_profiles": len(target_group_sizes),
        "later_duplicate_target_rows": int(y_data.duplicated().sum()),
        "repeated_target_groups": int((target_group_sizes > 1).sum()),
        "rows_in_repeated_target_groups": int(target_group_sizes[target_group_sizes > 1].sum()),
        "largest_repeated_target_group": int(target_group_sizes.max()),
        "longest_consecutive_target_run": run_length,
        "longest_run_start_zero_based": run_start,
        "longest_run_end_zero_based": run_end,
        "all_zero_target_rows": int(zero_target_mask.sum()),
        "all_zero_target_indices_zero_based": ",".join(
            str(index) for index in y_data.index[zero_target_mask]
        ),
    }


def main():
    AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loaded_data = {}
    file_summaries = []
    feature_rows = []

    # audit every source before any transformations are created
    for file_label, path in FILES.items():
        data, summary, duplicate_columns = audit_file(file_label, path)
        loaded_data[file_label] = data
        file_summaries.append(summary)

        if file_label.startswith("X_"):
            process = file_label.split("_")[1]
            feature_rows.extend(build_feature_quality(process, data, duplicate_columns))

    process_summaries = []
    for process in ["AlCu", "WTi"]:
        process_summaries.append(
            audit_process(
                process,
                loaded_data[f"X_{process}"],
                loaded_data[f"Y_{process}"],
            )
        )

    file_summary = pd.DataFrame(file_summaries)
    process_summary = pd.DataFrame(process_summaries)
    feature_quality = pd.DataFrame(feature_rows)

    file_summary.to_csv(AUDIT_RESULTS_DIR / "data_audit_files.csv", index=False)
    process_summary.to_csv(
        AUDIT_RESULTS_DIR / "data_audit_alignment.csv",
        index=False,
    )
    feature_quality.to_csv(
        AUDIT_RESULTS_DIR / "data_audit_features.csv",
        index=False,
    )

    print("Data audit complete.")
    print(file_summary[["dataset", "rows", "columns", "missing_cells", "official_checksum_match"]].to_string(index=False))
    print()
    print(process_summary[["process", "row_counts_match", "later_duplicate_target_rows", "all_zero_target_rows"]].to_string(index=False))
    print()

    flagged_features = feature_quality[
        feature_quality["dominant_value_80_percent"]
        | feature_quality["constant"]
        | feature_quality["duplicate_of"].ne("")
    ]
    print("Features requiring review because one value occurs in at least 80% of rows:")
    if flagged_features.empty:
        print("None")
    else:
        print(
            flagged_features[
                ["process", "feature", "unique_values", "modal_share", "zero_share"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
