# Multivariate PVD Process Monitoring and Virtual Metrology

**Release:** v1.0

This project uses anonymized PVD process measurements to build a retrospective
process-state monitoring and virtual-metrology workflow. AlCu is the primary
analysis. WTi repeats the same workflow with models fitted only on WTi data.

The project is meant to answer practical process-engineering questions: Can a
large set of correlated measurements be reduced to a smaller monitoring view?
Which anonymous channels should be reviewed first after a statistical
excursion? How well can the process inputs predict all 17 released target
measurements?

## Data

The source is Infineon's public
[Advanced Process Control and Statistical Process Control Data for Thickness Prediction of AlCu and WTi Metal Layer in Semiconductor Manufacturing](https://doi.org/10.5281/zenodo.16881338).

| Process | Input columns | Released targets | Rows |
|---|---:|---:|---:|
| AlCu | 97 | 17 | 4,848 |
| WTi | 108 | 17 | 1,740 |

The four local CSV files are unchanged copies of the published release. They
match the official MD5 checksums and are ignored by Git. The dataset is licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); its authors are
Amina Mević, Andreas Laber, and Senka Krivić. Download instructions and
checksums are in [`data/raw/README.md`](data/raw/README.md).

The release does not provide physical units, inverse scaling, timestamps,
equipment identifiers, measurement-point coordinates, specifications, or fault
labels. I therefore report released-scale quantities only. The derived mean
index and reference-profile deviation are engineering summaries, not physical
thickness or physical uniformity.

## Workflow

1. Check file identity, shapes, missing values, duplicates, and X/Y row counts.
2. Use a fixed 80/20 split as the primary validation. Run a second split grouped
   by exactly equal target profiles as a sensitivity check. Equal target
   profiles are not assumed to be duplicate wafers.
3. Keep each all-zero target record in process-state monitoring, exclude it from
   primary Y-based modeling, and run explicit with/without sensitivity checks.
4. Fit training-only standardization and PCA. Retain the smallest PCA model that
   reaches 90% explained variance, then calculate Hotelling T² and Q/SPE.
5. Use the training-set 99th percentiles as empirical alert thresholds and rank
   channel contributions for selected excursions.
6. Predict the 17 released targets directly with multi-output Linear Regression
   and Ridge. Add one fixed Random Forest because it passes a predeclared
   training-only improvement gate. Each split has its own gate; only the primary
   AlCu gate transfers the Random Forest configuration to WTi.

## Main results

| Result | AlCu | WTi |
|---|---:|---:|
| PCA components retained | 34 | 35 |
| Variance retained | 90.55% | 90.16% |
| Ordinary assessment alert rate | 1.13% | 4.31% |
| Random Forest 17-target MAE | 0.00615 | 0.00519 |
| Random Forest 17-target RMSE | 0.00870 | 0.00829 |
| Random Forest aggregate R² | 0.622 | 0.604 |
| Grouped-sensitivity Random Forest RMSE | 0.00926 | 0.00767 |

The AlCu grouped split increased Random Forest RMSE by 6.4% and reduced
aggregate R² from 0.622 to 0.543. The ordinary split is somewhat optimistic,
but the grouped result still contains useful released-scale predictive signal.
The WTi grouped result improved instead of degrading. AlCu and WTi RMSE values
belong to different released scales, so they are not a physical ranking of the
two processes.

For AlCu, the 11 valid alerted assessment rows had a released mean-index
difference of -0.00610 from the 959 normal rows. The clustered 95% interval was
[-0.01355, 0.00071]. The reference-profile deviation difference was +0.00270
with an interval of [-0.00111, 0.00643]. Both intervals cross zero, so this
dataset does not establish a clear released-output shift for alerted rows.

The primary AlCu Random Forest had mean row MAE of 0.00611 for normal rows and
0.01263 for both-alarm rows. The both-alarm group contains only four rows. This
is a useful reliability check, but it is too small to support a production rule.
Aggregate and per-target model errors are listed in the
[`results/` guide](results/README.md).

## Engineering figures

![AlCu retrospective process-state map](figures/05_alcu_process_state_map.png)

T² measures distance within the retained PCA space. Q/SPE measures variation
that the retained PCA model does not reconstruct well. The categories form a
review queue; they are not fault labels.

![AlCu excursion contribution plots](figures/06_alcu_excursion_contributions.png)

Contribution plots shorten the channel-review list for selected excursions.
They do not prove physical causation because the sensor names and equipment
context are unavailable.

![AlCu direct 17-target predictions](figures/08_alcu_virtual_metrology_predictions.png)

![AlCu and separately fitted WTi comparison](figures/13_cross_process_workflow_comparison.png)

## Run the analysis

Place the four downloaded CSV files in `data/raw/`, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR=.matplotlib-cache
python src/01_data_audit.py
python src/02_preprocess_and_quality.py
python src/03_pca_monitoring.py
python src/04_excursion_analysis.py
python src/05_virtual_metrology.py
python src/06_wti_validation.py
```

The scripts are numbered because later stages read outputs from earlier ones.
There is no notebook, dashboard, or package framework.

```text
data/raw/           local unchanged source CSVs, ignored by Git
data/processed/     generated splits, scores, and row-level predictions
src/                six analysis scripts
figures/            generated figures
results/audit/      checks and split summaries
results/alcu/       primary analysis tables
results/wti/        separately fitted WTi tables
results/comparison/ cross-process workflow summary
report/             concise technical report
```

## Limits on interpretation

- X/Y pairing is positional because the release contains no join key.
- There is no chronology or known-good baseline. The work is retrospective
  multivariate process-state monitoring, not formal chronological SPC.
- The 99th-percentile cutoffs are empirical alert thresholds, not validated
  control limits or independently calibrated false-alarm limits.
- There are no specifications for Cp or Cpk and no fault labels for validated
  fault detection.
- One all-zero target row occurs in each process, and its meaning is unknown.
- WTi is a second released process dataset, not an external fab qualification.

See [`report/technical_report.md`](report/technical_report.md) for the technical
discussion and [`results/README.md`](results/README.md) for the result tables.
