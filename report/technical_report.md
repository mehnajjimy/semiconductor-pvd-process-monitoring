# Multivariate PVD Process Monitoring and Virtual Metrology

**Technical report, release v1.0**

## 1. Scope and data

This project tests whether anonymized PVD process measurements can support a
compact process-state view and direct prediction of 17 released target
measurements. AlCu is the primary analysis. WTi repeats the workflow with models
fitted only on WTi data.

The data come from Infineon's public
[PVD APC/SPC dataset](https://doi.org/10.5281/zenodo.16881338) and its
[associated paper](https://doi.org/10.3233/FAIA251482). My four local CSV files
are unchanged copies of the release and match its published MD5 checksums. The
dataset authors are Amina Mević, Andreas Laber, and Senka Krivić, and the files
use the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

| Process | Inputs | Released targets | Rows |
|---|---:|---:|---:|
| AlCu | 97 | 17 | 4,848 |
| WTi | 108 | 17 | 1,740 |

The audit found no missing or nonfinite cells, exact duplicate X rows, duplicate
X columns, or exact duplicate combined X/Y rows. X/Y pairing is positional
because the release has no join key.

Physical units, inverse scaling, point coordinates, timestamps, equipment
identifiers, specifications, and fault labels are unavailable. All results stay
on the released scale. The mean index and reference-profile deviation used
below are statistical summaries, not physical thickness or physical uniformity.

## 2. Validation decisions

The primary validation is a fixed 80/20 random split using seed 42. A second
split keeps exactly equal 17-target profiles together. It is a sensitivity
check, not the primary result. Equal Y profiles are not assumed to be duplicate
wafers even though 1,203 AlCu rows and 93 WTi rows repeat an earlier profile.

Training-only feature screening removes `Sensor_26` and `Sensor_63` from AlCu
because one value occurs in at least 98% of the training rows. The remaining 95
AlCu inputs and all 108 WTi inputs are retained.

Each process has one all-zero target row: zero-based AlCu index 2,586 and WTi
index 936. Their meaning is unknown. They remain in X-based process-state
monitoring, are excluded from primary Y summaries and models, and are added back
only for explicit sensitivity checks.

## 3. Process-state monitoring

Inputs are standardized using ordinary training rows. PCA is then fit on those
rows, retaining the smallest model that reaches 90% explained variance.

| Process | Model inputs | Retained PCs | Variance retained |
|---|---:|---:|---:|
| AlCu | 95 | 34 | 90.55% |
| WTi | 108 | 35 | 90.16% |

Hotelling T² measures distance within the retained PCA space. Q/SPE measures
the squared variation that the retained model does not reconstruct. Their
thresholds are the 99th percentiles of the ordinary training statistics.

| Process | T² threshold | Q threshold | Assessment alerts |
|---|---:|---:|---:|
| AlCu | 258.819 | 60.379 | 11/970 (1.13%) |
| WTi | 419.905 | 60.568 | 15/348 (4.31%) |

These are empirical alert thresholds for retrospective multivariate
process-state monitoring. They are not formal chronological SPC control limits
or validated fault rules.

Three AlCu examples were selected by taking the largest threshold ratio from
each non-normal category:

| Row | Role | State | T²/threshold | Q/threshold |
|---:|---|---|---:|---:|
| 1,113 | training diagnostic | T²-only | 14.70 | 0.14 |
| 6 | training diagnostic | Q-only | 0.44 | 3.77 |
| 139 | held-out diagnostic | both-alarm | 180.71 | 1,568.71 |

Contribution terms point to channels for review. `Sensor_53`, `Sensor_56`, and
`Sensor_73` lead the first case; `Sensor_13`, `Sensor_97`, and `Sensor_46` lead
the second. `Sensor_35` and `Sensor_54` rank prominently for the held-out case.
These rankings are diagnostic and do not prove physical root cause.

For the valid AlCu assessment rows, exact target-profile groups were resampled
together in a 2,000-sample bootstrap:

| Released-scale summary | Alert minus normal mean | Clustered 95% interval |
|---|---:|---:|
| Mean index | -0.00610 | [-0.01355, 0.00071] |
| Reference-profile deviation | +0.00270 | [-0.00111, 0.00643] |

Both intervals include zero. The 11 alerted rows do not establish a clear
released-output shift.

## 4. Direct virtual metrology

The primary prediction task estimates all 17 released targets directly. Linear
Regression and Ridge are required baselines. Ridge tuning uses training-only
five-fold cross-validation with scaling inside each fold.

One Random Forest is retained: 300 trees, minimum leaf size 5, 70% of features
per split, and seed 42. Seeds 11, 42, and 73 improve ordinary training-fold RMSE
over Ridge by 39.0% to 39.2%. This ordinary gate controls the primary AlCu model
and the configuration passed to WTi. A separate grouped-training gate controls
only the grouped AlCu result. Neither held-out set affects the other gate.

| Process | Model | MAE | RMSE | Aggregate R² |
|---|---|---:|---:|---:|
| AlCu | Linear Regression | 0.01030 | 0.10607 | -55.257 |
| AlCu | Ridge | 0.01118 | 0.01486 | 0.001 |
| AlCu | Random Forest | 0.00615 | 0.00870 | 0.622 |
| WTi | Linear Regression | 0.01819 | 0.18202 | -176.846 |
| WTi | Ridge | 0.00721 | 0.01458 | -0.172 |
| WTi | Random Forest | 0.00519 | 0.00829 | 0.604 |

Negative R² means the model performs worse than a target-wise mean benchmark.
Rare large errors drive the poor Linear Regression RMSE. Random Forest gives a
clearer and more stable result.

The grouped sensitivity changes AlCu Random Forest RMSE from 0.00870 to 0.00926
and R² from 0.622 to 0.543. WTi changes from 0.00829 and 0.604 to 0.00767 and
0.690. Repeated profiles add some optimism to the ordinary AlCu score, but the
grouped result retains useful released-scale signal.

### Random Forest error by target

| Target | AlCu MAE | AlCu RMSE | WTi MAE | WTi RMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.00792 | 0.01143 | 0.00455 | 0.00693 |
| 2 | 0.00599 | 0.00858 | 0.00704 | 0.01096 |
| 3 | 0.00793 | 0.01085 | 0.00457 | 0.00722 |
| 4 | 0.00624 | 0.00876 | 0.00413 | 0.00670 |
| 5 | 0.00623 | 0.00860 | 0.00645 | 0.01031 |
| 6 | 0.00423 | 0.00580 | 0.00442 | 0.00715 |
| 7 | 0.00726 | 0.00982 | 0.00335 | 0.00534 |
| 8 | 0.00576 | 0.00782 | 0.00553 | 0.00875 |
| 9 | 0.00696 | 0.00960 | 0.00372 | 0.00592 |
| 10 | 0.00472 | 0.00649 | 0.00471 | 0.00755 |
| 11 | 0.00562 | 0.00788 | 0.00417 | 0.00649 |
| 12 | 0.00390 | 0.00545 | 0.00669 | 0.01023 |
| 13 | 0.00613 | 0.00857 | 0.00394 | 0.00603 |
| 14 | 0.00551 | 0.00745 | 0.00662 | 0.00988 |
| 15 | 0.00783 | 0.01080 | 0.00616 | 0.00927 |
| 16 | 0.00690 | 0.00975 | 0.00653 | 0.01010 |
| 17 | 0.00537 | 0.00744 | 0.00573 | 0.00886 |

Full-precision MAE, RMSE, and R² values are in the per-target CSV files listed
in the [`results/` guide](../results/README.md).

For AlCu, mean row MAE rises from 0.00611 for normal rows to 0.01263 for the
four both-alarm rows. This is worth reviewing, but the subgroup is too small for
a production reliability rule.

## 5. WTi replication and zero sensitivity

The WTi scaler, PCA model, thresholds, Ridge penalty, and final fits use WTi
training data. The script verifies that its Random Forest configuration matches
the model admitted by the ordinary AlCu training-only gate before refitting it.

WTi's 14 alerted valid assessment rows have a mean-index difference of +0.00186
from normal rows, with a clustered 95% interval of [-0.00420, 0.00768]. Its
reference-profile deviation interval also crosses zero.

| Process and split | Zero location | RF RMSE excluded | RF RMSE included |
|---|---|---:|---:|
| AlCu ordinary | training | 0.008698 | 0.008715 |
| AlCu grouped | assessment | 0.009258 | 0.020967 |
| WTi ordinary | assessment | 0.008285 | 0.041549 |
| WTi grouped | training | 0.007666 | 0.008124 |

The unknown row has its largest effect when it appears in assessment. This
supports excluding it from primary Y-based evaluation without assigning it a
meaning.

## 6. Limits and conclusion

- Anonymous variables cannot support named equipment diagnoses.
- No chronology or known-good period supports formal SPC limits.
- No specifications support Cp or Cpk, and no fault labels support validated
  fault detection.
- AlCu and WTi use different released scales, so their raw errors are not a
  physical cross-process ranking.
- WTi is a second public dataset, not an external fab qualification.

The project produces a reproducible retrospective monitoring and
virtual-metrology workflow. PCA reduces the process inputs, T² and Q/SPE sort
statistical excursions, and contribution plots prioritize channels for review.
The Random Forest keeps useful 17-target predictive signal under both split
designs. The results do not establish physical metrology, production control
limits, validated faults, or production readiness.
