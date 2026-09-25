# Multivariate PVD Process Monitoring and Virtual Metrology

**Release:** v1.0

This project uses anonymized physical vapor deposition (PVD) process data for
two things. One is a retrospective view of the multivariate process state. The
other is virtual metrology, which predicts the measured targets from the
process inputs. AlCu is the main analysis. WTi repeats the same workflow with
its own separately fitted models.

The workflow shrinks many correlated process inputs into a smaller monitoring
view. It ranks anonymous channels so excursions can be reviewed in order. It
also predicts all 17 released target measurements directly.

## Data

The data come from Infineon's public
[Advanced Process Control and Statistical Process Control Data for Thickness Prediction of AlCu and WTi Metal Layer in Semiconductor Manufacturing](https://doi.org/10.5281/zenodo.16881338).

| Process | Input columns | Released targets | Rows |
|---|---:|---:|---:|
| AlCu | 97 | 17 | 4,848 |
| WTi | 108 | 17 | 1,740 |

My four local CSV files are unchanged copies of the release, and they match its
MD5 checksums. Git ignores them. Amina Mević, Andreas Laber, and Senka Krivić
published the dataset under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Download steps and
checksums are in [`data/raw/README.md`](data/raw/README.md).

The release leaves out physical units, inverse scaling, timestamps, equipment
identifiers, measurement-point coordinates, specifications, and fault labels.
So every number here stays on the released scale. The mean index and
reference-profile deviation are summaries I derived for engineering review.
They are not physical thickness or physical uniformity.

## Method

- Check file identity, shapes, missing values, duplicates, and X/Y row counts.
- Validate on a fixed 80/20 split first. Then validate again with rows that
  share exactly the same target profile kept together. I do not assume that
  equal profiles are duplicate wafers.
- Keep the all-zero target records for process-state monitoring. Leave them
  out of the main Y-based models, and run checks with and without them. Their
  meaning is unknown.
- Fit standardization and PCA on training data only. Keep the fewest
  components that explain at least 90% of the variance. Then compute Hotelling
  T² and Q/SPE, with empirical alert thresholds at the 99th percentile of the
  training set.
- Predict the 17 targets with multi-output Linear Regression and Ridge. A fixed
  Random Forest is also included because it passed an improvement gate that
  was set in advance and uses training data only. Each validation split has its
  own gate. Only the primary AlCu gate passes the model settings on to WTi.

## Results

| Result | AlCu | WTi |
|---|---:|---:|
| PCA components retained | 34 | 35 |
| Variance retained | 90.55% | 90.16% |
| Ordinary assessment alert rate | 1.13% | 4.31% |
| Random Forest 17-target MAE | 0.00615 | 0.00519 |
| Random Forest 17-target RMSE | 0.00870 | 0.00829 |
| Random Forest aggregate R² | 0.622 | 0.604 |
| Grouped-sensitivity Random Forest RMSE | 0.00926 | 0.00767 |

For AlCu, keeping identical target profiles together raised Random Forest RMSE
by 6.4% and lowered aggregate R² from 0.622 to 0.543. So the ordinary split is
a little optimistic. Even so, the grouped result still predicts the
released-scale targets usefully. WTi did better under the grouped split. The
two processes use different released scales, so their errors do not rank them
physically.

For AlCu, the alerted rows did not show a clear shift in released output
compared with normal rows. Both clustered 95% intervals crossed zero. Only four
assessment rows set off both the T² and Q/SPE alarms. Their prediction error was
larger, but that is a description, not a production rule. The
[`results/` guide](results/README.md) lists aggregate and per-target errors.

## Engineering figures

![AlCu retrospective process-state map](figures/05_alcu_process_state_map.png)

T² and Q/SPE flag statistical excursions in different ways. The categories set
review priority. They are not fault labels.

![AlCu excursion contribution plots](figures/06_alcu_excursion_contributions.png)

Contribution plots show which anonymous channels to review first. They do not
prove physical cause.

![AlCu direct 17-target virtual metrology predictions](figures/08_alcu_virtual_metrology_predictions.png)

The held-out predictions cover all 17 released targets directly. The mean index
appears only as a secondary engineering summary.

## Reproduce

Put the four downloaded CSV files in `data/raw/`, then run:

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

Run the scripts in this order, because later ones read what earlier ones write.

## Limits

- X/Y pairing goes by row position, because the release has no join key.
- There is no chronology or known-good baseline. So this is retrospective
  multivariate process-state monitoring, not formal chronological SPC.
- The thresholds are empirical alerts. They are not validated control limits,
  and their false-alarm rates are not independently calibrated.
- Without specifications there is no Cp/Cpk analysis. Without fault labels
  there is no validated fault detection.
- WTi is a second released process dataset, not an external fab qualification.

The [`technical report`](report/technical_report.md) has the full method and
interpretation. The [`results/` guide](results/README.md) covers the main
tables and the supporting sensitivity outputs.
