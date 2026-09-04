# Multivariate PVD Process Monitoring and Virtual Metrology

## 1. Engineering Problem

PVD equipment can produce many correlated process measurements for each
wafer-level observation. Reviewing nearly 100 anonymous channels one at a time
is not a practical monitoring strategy. This project asks whether the released
measurements can be converted into a smaller retrospective process-state view,
whether selected statistical excursions can be traced back to a short list of
channels for engineering review, and whether the same process inputs contain
useful information about 17 released target measurements.

AlCu is the primary development analysis. WTi is a separately fitted
validation workflow. Machine learning supports the process-monitoring story
rather than replacing it.

## 2. Dataset

The analysis uses the attached copies of Infineon's public
[PVD APC/SPC dataset](https://doi.org/10.5281/zenodo.16881338) and its
[associated paper](https://doi.org/10.3233/FAIA251482). The release is licensed
CC BY 4.0. The authors describe each row as a wafer-level instance, the X files
as aggregated process-sensor statistics, and each Y file as 17 target
measurements related to layer thickness.

| Process | X shape | Y shape |
|---|---:|---:|
| AlCu | 4,848 × 97 | 4,848 × 17 |
| WTi | 1,740 × 108 | 1,740 × 17 |

The attached files match the official MD5 checksums. No replacement, synthetic,
or mirrored data were used.

The release does not include an observation key. X and Y have equal row counts
and are used in their released row order, but the positional pairing cannot be
independently checked against a stored identifier. Sensor identities, units,
inverse scaling, point coordinates, timestamps, recipes, tools, chambers,
fault labels, known-good labels, and engineering specification limits are also
unavailable.

## 3. Data Quality

All 205 process-input columns and all 34 target columns are numeric. There are
no missing or nonfinite cells, constant columns, exact duplicate X rows,
duplicate X columns, or exact duplicate combined X/Y rows.

Exact Y profiles repeat: 1,203 AlCu rows and 93 WTi rows repeat an earlier
17-value profile. Different X rows sharing one Y profile are not proven
duplicate wafers. The ordinary fixed split is therefore the primary
validation; a second split keeps equal target profiles together to measure
sensitivity to this structure.

Training-only screening removes AlCu `Sensor_26` and `Sensor_63` because one
value occurs in at least 98% of the primary training rows. The other 95 AlCu
features and all 108 WTi features are retained. Correlated features are not
aggressively removed because PCA is intended to represent their shared
variation.

One all-zero 17-target record occurs in each process: zero-based AlCu index
2,586 and WTi index 936. Their physical or data-system meaning is unknown. Both
remain in process-state inspection, are excluded from primary target summaries
and models, and are added back only for explicit sensitivity checks.

## 4. Released-Scale Output Summaries

The 17 public values do not have documented physical units or inverse scaling.
This report therefore does not call a derived value physical thickness or
physical uniformity.

The primary level summary is the arithmetic mean of the 17 released values,
called the **released-scale mean index**. Standard deviation, range, and
coefficient of variation are retained as descriptive released-scale
statistics.

Point baselines differ substantially. A secondary shape proxy first divides
each target by its point-specific median from valid primary training rows,
divides that vector by its own mean, and then calculates its root-mean-square
distance from one. This **reference-profile shape deviation** measures departure
from the typical released profile after removing row level. It is not a
physical uniformity measure, and it is undefined for an all-zero row.

| Process | Valid rows | Mean-index average | Shape-deviation average |
|---|---:|---:|---:|
| AlCu | 4,847 | 0.61250 | 0.01366 |
| WTi | 1,739 | 0.73446 | 0.00585 |

## 5. PCA Process Representation

Each process variable is standardized using only its ordinary training rows.
A full PCA is then fitted on those rows. The retained representation is the
smallest number of components reaching at least 90% cumulative explained
variance.

| Process | Model features | Retained PCs | Retained variance |
|---|---:|---:|---:|
| AlCu | 95 | 34 | 90.55% |
| WTi | 108 | 35 | 90.16% |

PCA reduces the number of directions an engineer must monitor while retaining
most released process-input variation. The components are statistical
combinations of anonymous channels, not named physical mechanisms.

## 6. T² and Q/SPE Monitoring

For a standardized observation `z`, retained PCA scores `t`, retained loading
matrix `P`, and retained eigenvalues `λ`:

- Hotelling T² is the sum of `t² / λ`. A high value means the observation is
  far from the center along retained process-variation directions.
- Q/SPE is `||z - tP||²`. A high value means the retained model does not
  reconstruct the observation's correlation pattern well.

The limits are the 99th percentiles of the ordinary training statistics:

| Process | T² threshold | Q threshold | Held-out any-alert rate |
|---|---:|---:|---:|
| AlCu | 258.819 | 60.379 | 1.13% (11/970) |
| WTi | 419.905 | 60.568 | 4.31% (15/348) |

Each observation is labeled normal, T²-only, Q-only, or both-alarm. These are
empirical alert thresholds for retrospective multivariate process-state
monitoring. The data do not establish chronology or a known-good baseline, so
the thresholds are not formal chronological SPC control limits and the alerts
are not validated faults. The 99th percentiles use the same training rows that
fit the scaler and PCA, so they are descriptive historical cutoffs rather than
independently calibrated false-alarm limits.

## 7. Selected AlCu Excursions

Three nonzero-output observations were selected to make the diagnostic review
focused:

| Zero-based index | Split | Category | T²/limit | Q/limit |
|---:|---|---|---:|---:|
| 1,113 | Training | T²-only | 14.70 | 0.14 |
| 6 | Training | Q-only | 0.44 | 3.77 |
| 139 | Held-out | Both-alarm | 180.71 | 1,568.71 |

The first case is unusual primarily inside the retained PCA space, the second
primarily outside it, and the third in both senses.
The first two are training-set diagnostics; only index 139 is an independent
held-out example.

## 8. Contribution Analysis

Q contributions are squared standardized reconstruction residuals and add
exactly to Q. Signed T² terms use the decomposition
`zᵢ[P'Λ⁻¹Pz]ᵢ` and add exactly to T². Because channels are correlated, the T²
decomposition is diagnostic rather than a unique causal attribution.

The leading channels were:

- T²-only index 1,113: `Sensor_53`, `Sensor_56`, `Sensor_73`,
  `Sensor_52`, and `Sensor_70`.
- Q-only index 6: `Sensor_13`, `Sensor_97`, `Sensor_46`,
  `Sensor_38`, and `Sensor_80`.
- Both-alarm index 139: `Sensor_35` and `Sensor_54` appear prominently
  in both T² and Q rankings.

These are measurement channels contributing strongly to statistical
excursions and candidates for engineering investigation. Anonymous data cannot
establish a physical root cause.

## 9. Excursions Versus Released Outputs

The AlCu ordinary held-out set contains 959 normal and 11 any-alert
observations after excluding the all-zero target record. Exact target-profile
groups are resampled together in a 2,000-sample clustered bootstrap.

| Metric | Alert minus normal mean | Clustered 95% interval |
|---|---:|---:|
| Released-scale mean index | -0.00610 | [-0.01355, 0.00072] |
| Reference-profile shape deviation | +0.00270 | [-0.00111, 0.00643] |

Both intervals include zero. The held-out data therefore do not establish a
clear difference in either released-scale summary. The small alert count also
limits precision. No positive quality story is forced from these results.

Including the AlCu all-zero record in the full descriptive mean-index
comparison changes the any-alert minus normal difference from -0.00059 to
-0.00864. This sensitivity illustrates why the unknown record is handled
explicitly; it does not assign that record a physical interpretation.

## 10. Direct 17-Target Virtual Metrology

The primary prediction task is all 17 targets directly. Multi-output Linear
Regression and Ridge are required baselines. Ridge penalties are selected by
five-fold training-only cross-validation with scaling fitted inside each fold.
Rare high-leverage process states make unregularized predictions unstable and
drive Ridge toward strong regularization.

One Random Forest configuration is admitted only because it exceeds a
predeclared 10% RMSE-improvement gate over Ridge in training-only folds for
both AlCu split designs and seeds 11, 42, and 73. Improvement ranges from 39.0%
to 39.2% for the ordinary design and 34.0% to 34.2% for the grouped design.
Held-out rows are not used for admission. No broad nonlinear model or
hyperparameter search is performed.

### AlCu ordinary fixed split

These metrics are conditional on excluding the unknown all-zero target row.

| Model | MAE | RMSE | Aggregate R² |
|---|---:|---:|---:|
| Linear Regression | 0.01031 | 0.10607 | -55.257 |
| Ridge | 0.01118 | 0.01486 | 0.001 |
| Random Forest | 0.00615 | 0.00870 | 0.622 |

Linear Regression's MAE hides rare very large errors, which are exposed by
RMSE and R². Random Forest per-target R² ranges from 0.517 to 0.823.

The grouped-profile sensitivity contains no target-profile overlap. Random
Forest RMSE increases 6.4%, from 0.00870 to 0.00926, and aggregate R² changes
from 0.622 to 0.543. Repeated target profiles add some optimism to the
ordinary split, but they do not account for the complete released-scale
predictive relationship.

## 11. Model Reliability During Excursions

For the primary AlCu Random Forest, mean row MAE is:

| Retrospective state | Rows | Mean row MAE |
|---|---:|---:|
| Normal | 959 | 0.00612 |
| T²-only | 5 | 0.00644 |
| Q-only | 2 | 0.00795 |
| Both-alarm | 4 | 0.01263 |

The both-alarm mean is about twice the normal mean, which is consistent with a
model being less reliable for some unusual process states. The subgroup has
only four observations, so this is a descriptive diagnostic rather than a
validated reliability rule.

## 12. Separately Fitted WTi Validation

The WTi model is fitted from WTi training rows; an AlCu-fitted estimator is not
applied to WTi. The Random Forest structure is fixed after the AlCu gate rather
than selected from WTi assessment results.

| WTi validation | MAE | RMSE | Aggregate R² |
|---|---:|---:|---:|
| Ordinary fixed split | 0.00519 | 0.00829 | 0.604 |
| Grouped-profile sensitivity | 0.00513 | 0.00767 | 0.690 |

WTi per-target R² ranges from 0.488 to 0.758. Its grouped result improves rather
than degrades, so repeated target profiles do not appear to inflate the
ordinary WTi result. PCA and T²/Q also form a compact process-state view, but
WTi's held-out alert rate is higher than AlCu's. This supports the workflow's
use on a second released process while showing that monitoring behavior is
process-specific.
The WTi metrics are also conditional on the stated all-zero exclusion. Raw
AlCu and WTi RMSE values use process-specific released scales and are not a
physical cross-process ranking.

For WTi's 333 normal and 14 any-alert valid held-out rows, the alert-minus-
normal released mean-index difference is +0.00186 with a clustered 95%
interval of [-0.00420, 0.00768]. The shape-deviation difference is +0.00084
with an interval of [-0.00137, 0.00405]. As with AlCu, neither comparison
establishes a clear released-output shift.

WTi Q-only and both-alarm rows have higher mean Random Forest errors than
normal rows, but the groups contain seven and five observations. The pattern
remains descriptive.

## 13. All-Zero Modeling Sensitivity

The all-zero rows are excluded from primary Y-based modeling.

- AlCu's zero row lies in the ordinary training set. Adding it changes Random
  Forest assessment RMSE only from 0.008698 to 0.008715.
- In the grouped AlCu split, the zero row lies in assessment; adding that
  single row changes aggregate RMSE from 0.009258 to 0.020967.
- WTi's zero row lies in ordinary assessment; adding it changes Random Forest
  aggregate RMSE from 0.008285 to 0.041549.
- In the grouped WTi split, it lies in training; adding it changes Random
  Forest assessment RMSE from 0.007666 to 0.008124.

These changes report sensitivity to an unusual released record. They do not
identify why the record is zero.

## 14. Limitations

- The public variables are anonymized and cannot support named sensor or
  equipment diagnoses.
- Released values have no documented units or inverse scaling, so results are
  not physical thickness or physical uniformity.
- The 17 point coordinates are unavailable; no center-to-edge metric is used.
- No join key independently verifies positional X/Y pairing.
- No timestamps establish chronological order.
- No known-good reference period establishes formal SPC control limits.
- PCA fitting and empirical threshold estimation share the same training rows,
  so alert rates are not independently calibrated false-alarm rates.
- No USL or LSL supports Cp or Cpk.
- No fault labels support validated fault-detection claims.
- Exact target profiles repeat, and the grouped sensitivity is still only one
  alternate split.
- Alert subgroups are small, and statistical association is not causation.
- WTi is a second public process dataset, not an external production
  qualification.

## 15. Conclusions

PCA reduced 95 AlCu modeling channels to 34 process-variation directions and
108 WTi channels to 35 while retaining at least 90% variance. Explicit T² and
Q/SPE calculations separated retained-space distance from residual-space
behavior and produced practical retrospective investigation queues.
Contribution analysis then reduced selected alerts to short anonymous channel
lists without claiming root cause.

The AlCu held-out data did not show a clear association between statistical
alerts and the two released-scale output summaries. Direct virtual metrology
was more successful: one gated Random Forest predicted the 17 released targets
with aggregate R² of 0.622 for AlCu and 0.604 for separately fitted WTi.
Grouped-profile validation preserved useful performance. Higher model errors
for some both-alarm observations are an engineering reason to review
prediction reliability alongside process state, but the small groups do not
support a production rule.

The appropriate claim is a reproducible, retrospective demonstration of
multivariate process-state monitoring and released-scale virtual metrology—not
formal chronological SPC, validated fault detection, physical metrology, or a
production-ready controller.
