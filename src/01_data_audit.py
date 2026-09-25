from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd


# folders and the four published csv files
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
AUDIT_RESULTS_DIR = PROJECT_DIR / "results" / "audit"

FILES = {
    "X_AlCu": RAW_DIR / "X_pvd_AlCu.csv",
    "Y_AlCu": RAW_DIR / "Y_pvd_AlCu.csv",
    "X_WTi": RAW_DIR / "X_pvd_WTi.csv",
    "Y_WTi": RAW_DIR / "Y_pvd_WTi.csv",
}

# checksums published with the zenodo record
EXPECTED_MD5 = {
    "X_AlCu": "9dd4bfc44ffb6a570e519000e07e9fe8",
    "Y_AlCu": "99b845acd32ae57060e194e407c08f19",
    "X_WTi": "cfc27ea3bdf780bcdf6dfdc5711836c1",
    "Y_WTi": "ab6a3b425d18d6f88f0466a7c40b340a",
}

CHUNK_SIZE = 1024 * 1024


# file level checks


# read the file in chunks so the checksum works without loading the csv twice
def calculate_md5(path):
    file_hash = md5()
    with path.open("rb") as source_file:
        while True:
            chunk = source_file.read(CHUNK_SIZE)
            if chunk == b"":
                break
            file_hash.update(chunk)
    return file_hash.hexdigest()


# compare full columns because equal correlations do not prove exact duplication
# each duplicate points to the first earlier column it matches
def find_duplicate_columns(data):
    duplicates = {}
    columns = list(data.columns)
    for column_number, column in enumerate(columns):
        for earlier_column in columns[:column_number]:
            if data[column].equals(data[earlier_column]):
                duplicates[column] = earlier_column
                break
    return duplicates


# keep the audit read-only and calculate every check from the published file
def audit_file(file_label, path):
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


# feature and alignment checks


# modal share highlights rare-state channels that standardization can overemphasize
def build_feature_quality(process, data, duplicate_columns):
    rows = []
    for column in data.columns:
        value_counts = data[column].value_counts(dropna=False)
        modal_count = int(value_counts.iloc[0])
        modal_value = value_counts.index[0]
        unique_values = data[column].nunique(dropna=False)
        modal_share = modal_count / len(data)
        rows.append(
            {
                "process": process,
                "feature": column,
                "unique_values": int(unique_values),
                "modal_value": float(modal_value),
                "modal_count": modal_count,
                "modal_share": modal_share,
                "zero_share": float((data[column] == 0).mean()),
                "mean": float(data[column].mean()),
                "standard_deviation": float(data[column].std(ddof=0)),
                "minimum": float(data[column].min()),
                "maximum": float(data[column].max()),
                "constant": bool(unique_values <= 1),
                "near_constant_98_percent": bool(modal_share >= 0.98),
                "dominant_value_80_percent": bool(modal_share >= 0.80),
                "duplicate_of": duplicate_columns.get(column, ""),
            }
        )
    return rows


# consecutive repeated targets can reveal grouping that a random split may separate
# returns the zero-based start, end and length of the longest run of equal rows
# the earliest run wins a tie
def longest_identical_run(data):
    longest_start = 0
    longest_end = 0
    current_start = 0

    for row_number in range(1, len(data)):
        row_changed = not data.iloc[row_number].equals(data.iloc[row_number - 1])
        if row_changed:
            current_length = row_number - current_start
            longest_length = longest_end - longest_start + 1
            if current_length > longest_length:
                longest_start = current_start
                longest_end = row_number - 1
            current_start = row_number

    # the last run has no change after it, so check it here
    final_length = len(data) - current_start
    longest_length = longest_end - longest_start + 1
    if final_length > longest_length:
        longest_start = current_start
        longest_end = len(data) - 1

    return longest_start, longest_end, longest_end - longest_start + 1


# x and y have no stored key, so only positional pairing can be checked
def audit_process(process, x_data, y_data):
    combined = pd.concat([x_data, y_data], axis=1)
    zero_target_mask = (y_data == 0).all(axis=1)
    target_group_sizes = y_data.groupby(
        list(y_data.columns),
        dropna=False,
        sort=False,
    ).size()
    repeated_groups = target_group_sizes > 1
    run_start, run_end, run_length = longest_identical_run(y_data)
    zero_target_indices = [str(index) for index in y_data.index[zero_target_mask]]

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
        "repeated_target_groups": int(repeated_groups.sum()),
        "rows_in_repeated_target_groups": int(target_group_sizes[repeated_groups].sum()),
        "largest_repeated_target_group": int(target_group_sizes.max()),
        "longest_consecutive_target_run": run_length,
        "longest_run_start_zero_based": run_start,
        "longest_run_end_zero_based": run_end,
        "all_zero_target_rows": int(zero_target_mask.sum()),
        "all_zero_target_indices_zero_based": ",".join(zero_target_indices),
    }


# running the audit


# audit every source before any transformations are created
# feature quality rows come from the two input files only
def audit_all_files():
    loaded_data = {}
    file_summaries = []
    feature_rows = []

    for file_label, path in FILES.items():
        data, summary, duplicate_columns = audit_file(file_label, path)
        loaded_data[file_label] = data
        file_summaries.append(summary)

        if file_label.startswith("X_"):
            process = file_label.split("_")[1]
            feature_rows.extend(build_feature_quality(process, data, duplicate_columns))

    return loaded_data, file_summaries, feature_rows


# pair each input file with its target file
def audit_both_processes(loaded_data):
    process_summaries = []
    for process in ["AlCu", "WTi"]:
        x_data = loaded_data[f"X_{process}"]
        y_data = loaded_data[f"Y_{process}"]
        process_summaries.append(audit_process(process, x_data, y_data))
    return process_summaries


# write the three audit tables
def save_audit_tables(file_summary, process_summary, feature_quality):
    file_summary.to_csv(AUDIT_RESULTS_DIR / "data_audit_files.csv", index=False)
    process_summary.to_csv(AUDIT_RESULTS_DIR / "data_audit_alignment.csv", index=False)
    feature_quality.to_csv(AUDIT_RESULTS_DIR / "data_audit_features.csv", index=False)


# print a short summary and every feature that needs a closer look
def print_audit_report(file_summary, process_summary, feature_quality):
    file_columns = ["dataset", "rows", "columns", "missing_cells", "official_checksum_match"]
    process_columns = ["process", "row_counts_match", "later_duplicate_target_rows", "all_zero_target_rows"]
    flagged_columns = ["process", "feature", "unique_values", "modal_share", "zero_share"]

    print("Data audit complete.")
    print(file_summary[file_columns].to_string(index=False))
    print()
    print(process_summary[process_columns].to_string(index=False))
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
        print(flagged_features[flagged_columns].to_string(index=False))


# audit the files, save the tables, then print the summary
def main():
    AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loaded_data, file_summaries, feature_rows = audit_all_files()
    process_summaries = audit_both_processes(loaded_data)

    file_summary = pd.DataFrame(file_summaries)
    process_summary = pd.DataFrame(process_summaries)
    feature_quality = pd.DataFrame(feature_rows)

    save_audit_tables(file_summary, process_summary, feature_quality)
    print_audit_report(file_summary, process_summary, feature_quality)


if __name__ == "__main__":
    main()
