# Multivariate PVD Process Monitoring and Virtual Metrology

**Technical report, release v1.0**

## 1. Scope and data

This project asks whether anonymized PVD process measurements can do two jobs.
One is a compact view of the process state. The other is direct prediction of
17 released target measurements. AlCu is the main analysis. WTi repeats the
workflow with models fitted only on WTi data.

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

The audit found no missing or nonfinite cells, no exact duplicate X rows, no
duplicate X columns, and no exact duplicate combined X/Y rows. X/Y pairing goes
by row position because the release has no join key.

The release has no physical units, inverse scaling, point coordinates,
timestamps, equipment identifiers, specifications, or fault labels. So all
results stay on the released scale. The mean index and reference-profile
deviation used below are statistical summaries, not physical thickness or
physical uniformity.

## 2. Validation decisions

The main validation is a fixed 80/20 random split with seed 42. A second split
keeps rows with exactly the same 17-target profile together. It is a
sensitivity check, not the main result. I do not assume that equal Y profiles
are duplicate wafers, even though 1,203 AlCu rows and 93 WTi rows repeat an
earlier profile.

Feature screening on the training rows drops `Sensor_26` and `Sensor_63` from
AlCu, because a single value fills at least 98% of the training rows in each.
The other 95 AlCu inputs and all 108 WTi inputs stay in.

Each process has one row where every target is zero. These are zero-based
AlCu index 2,586 and WTi index 936. Their meaning is unknown. They stay in the
X-based process-state monitoring. They are left out of the main Y summaries and
models, and added back only for explicit sensitivity checks.

## 3. Process-state monitoring

Inputs are standardized using ordinary training rows. PCA is then fit on those
rows, keeping the smallest model that reaches 90% explained variance.

| Process | Model inputs | Retained PCs | Variance retained |
|---|---:|---:|---:|
| AlCu | 95 | 34 | 90.55% |
| WTi | 108 | 35 | 90.16% |

Hotelling T² measures distance inside the retained PCA space. Q/SPE measures
the squared variation that the retained model does not reconstruct. Each
threshold is the 99th percentile of that statistic on the ordinary training
rows.

| Process | T² threshold | Q threshold | Assessment alerts |
|---|---:|---:|---:|
| AlCu | 258.819 | 60.379 | 11/970 (1.13%) |
| WTi | 419.905 | 60.568 | 15/348 (4.31%) |

These are empirical alert thresholds for looking back at the multivariate
process state. They are not formal chronological SPC control limits or
validated fault rules.

I picked three AlCu examples by taking the row with the largest threshold
ratio in each non-normal category:

| Row | Role | State | T²/threshold | Q/threshold |
|---:|---|---|---:|---:|
| 1,113 | training diagnostic | T²-only | 14.70 | 0.14 |
| 6 | training diagnostic | Q-only | 0.44 | 3.77 |
| 139 | held-out diagnostic | both-alarm | 180.71 | 1,568.71 |

Contribution terms point to channels worth reviewing. `Sensor_53`,
`Sensor_56`, and `Sensor_73` lead the first case. `Sensor_13`, `Sensor_97`, and
`Sensor_46` lead the second. `Sensor_35` and `Sensor_54` rank high for the
held-out case. These rankings help diagnosis, but they do not prove a physical
root cause.

For the valid AlCu assessment rows, I ran a 2,000-sample bootstrap that
resampled exact target-profile groups together:

| Released-scale summary | Alert minus normal mean | Clustered 95% interval |
|---|---:|---:|
| Mean index | -0.00610 | [-0.01355, 0.00071] |
| Reference-profile deviation | +0.00270 | [-0.00111, 0.00643] |

Both intervals include zero. So the 11 alerted rows do not show a clear shift
in released output.

## 4. Direct virtual metrology

The main prediction task estimates all 17 released targets directly. Linear
Regression and Ridge are the required baselines. Ridge is tuned with five-fold
cross-validation on training data only, with scaling done inside each fold.

One Random Forest is kept. It uses 300 trees, minimum leaf size 5, 70% of
features per split, and seed 42. With seeds 11, 42, and 73 it improves ordinary
training-fold RMSE over Ridge by 39.0% to 39.2%. This ordinary gate decides the
main AlCu model and the settings passed to WTi. A separate gate on the grouped
training data decides only the grouped AlCu result. Neither held-out set
affects the other gate.

| Process | Model | MAE | RMSE | Aggregate R² |
|---|---|---:|---:|---:|
| AlCu | Linear Regression | 0.01030 | 0.10607 | -55.257 |
| AlCu | Ridge | 0.01118 | 0.01486 | 0.001 |
| AlCu | Random Forest | 0.00615 | 0.00870 | 0.622 |
| WTi | Linear Regression | 0.01819 | 0.18202 | -176.846 |
| WTi | Ridge | 0.00721 | 0.01458 | -0.172 |
| WTi | Random Forest | 0.00519 | 0.00829 | 0.604 |

A negative R² means the model does worse than simply predicting each target's
mean. A few rare large errors cause the poor Linear Regression RMSE. Random
Forest gives a clearer and steadier result.

Under the grouped split, AlCu Random Forest RMSE goes from 0.00870 to 0.00926
and R² goes from 0.622 to 0.543. WTi goes from 0.00829 and 0.604 to 0.00767 and
0.690. Repeated profiles make the ordinary AlCu score a bit optimistic, but the
grouped result still carries useful signal on the released scale.

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

For AlCu, mean row MAE rises from 0.00611 on normal rows to 0.01263 on the four
both-alarm rows. That is worth a look, but four rows are too few to set a
production reliability rule.

## 5. WTi replication and zero sensitivity

The WTi scaler, PCA model, thresholds, Ridge penalty, and final fits all use
WTi training data. Before refitting its Random Forest, the script checks that
the settings match the model that the ordinary AlCu training-only gate
accepted.

The 14 alerted valid WTi assessment rows differ from normal rows by +0.00186 in
mean index, with a clustered 95% interval of [-0.00420, 0.00768]. The interval
for reference-profile deviation also crosses zero.

| Process and split | Zero location | RF RMSE excluded | RF RMSE included |
|---|---|---:|---:|
| AlCu ordinary | training | 0.008698 | 0.008715 |
| AlCu grouped | assessment | 0.009258 | 0.020967 |
| WTi ordinary | assessment | 0.008285 | 0.041549 |
| WTi grouped | training | 0.007666 | 0.008124 |

The unknown all-zero row matters most when it lands in the assessment set.
That supports leaving it out of the main Y-based evaluation, without giving it
a meaning.

## 6. Limits and conclusion

- Anonymous variables cannot support diagnoses of named equipment.
- With no chronology or known-good period, formal SPC limits are not possible.
- With no specifications there is no Cp or Cpk. With no fault labels there is
  no validated fault detection.
- AlCu and WTi use different released scales, so their raw errors do not rank
  the processes physically.
- WTi is a second public dataset, not an external fab qualification.

The project gives a reproducible workflow for retrospective monitoring and
virtual metrology. PCA shrinks the process inputs. T² and Q/SPE sort
statistical excursions. Contribution plots rank channels for review. The Random
Forest keeps useful 17-target predictive signal under both split designs. The
results do not show physical metrology, production control limits, validated
faults, or production readiness.
