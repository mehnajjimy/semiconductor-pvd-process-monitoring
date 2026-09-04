# Multivariate PVD Process Monitoring and Virtual Metrology

**Technical report, release v1.0**

## 1. Purpose and data

PVD process records can contain many correlated measurements. Reviewing every
channel separately makes it harder to see the process state as a whole. I used
PCA, Hotelling T², Q/SPE, contribution analysis, and multi-output regression to
build one compact retrospective workflow. AlCu is the primary analysis. WTi is
a separately fitted replication on a second released process dataset.

The data come from Infineon's public
[PVD APC/SPC dataset](https://doi.org/10.5281/zenodo.16881338) and its
[associated paper](https://doi.org/10.3233/FAIA251482). The four CSV files in my
local analysis are unchanged copies of the release and match its published MD5
checksums. The dataset authors are Amina Mević, Andreas Laber, and Senka Krivić,
and the release uses the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

| Process | Input matrix | Target matrix |
|---|---:|---:|
| AlCu | 4,848 × 97 | 4,848 × 17 |
| WTi | 1,740 × 108 | 1,740 × 17 |

All columns are numeric. The audit found no missing or nonfinite values, exact
duplicate X rows, duplicate X columns, or exact duplicate combined X/Y rows.
The release has no join key, so X/Y pairing depends on the published row order.

Exact target profiles repeat. AlCu has 1,203 rows that repeat an earlier target
profile, while WTi has 93. Different X rows with an equal Y profile are not
proven duplicates. I kept a normal fixed 80/20 split as the primary validation
and used a grouped-by-exact-target-profile split only as a sensitivity check.

Training-only screening removed `Sensor_26` and `Sensor_63` from AlCu because
one value occurs in at least 98% of the primary training rows. The model keeps
the other 95 AlCu inputs and all 108 WTi inputs.

The release does not supply physical units, an inverse scale, measurement-point
coordinates, timestamps, tools, chambers, recipes, specifications, known-good
labels, or fault labels. Every result in this report stays on the released
scale. The mean index and reference-profile deviation are statistical summaries
and are not physical thickness or physical uniformity.

## 2. Validation and unknown zero records

The primary split uses random seed 42. All data-dependent preprocessing is fit
on the applicable training rows. Ridge tuning uses five-fold training-only
cross-validation with scaling inside each fold. The grouped sensitivity also
uses grouped folds, which keep exactly equal target profiles together.

One all-zero 17-target row occurs in each process: zero-based AlCu index 2,586
and WTi index 936. The dataset does not explain these records, so I do not assign
them a physical or operational meaning. Both remain in X-based process-state
monitoring. They are excluded from primary Y summaries and models, then added
back only for the reported sensitivity checks.

## 3. Retrospective multivariate monitoring

I standardized the retained process inputs with ordinary training-row means and
standard deviations. A full PCA was then fit on those rows, and I kept the
smallest number of components that reached 90% cumulative explained variance.

| Process | Model inputs | Retained PCs | Retained variance |
|---|---:|---:|---:|
| AlCu | 95 | 34 | 90.55% |
| WTi | 108 | 35 | 90.16% |

For PCA score vector `t` and retained eigenvalues `λ`, Hotelling T² is the sum
of `t²/λ`. It measures distance from the training center within the retained
PCA space. Q/SPE is the squared reconstruction residual. It measures process
variation that the retained space does not reconstruct well.

The alert thresholds are the 99th percentiles of the corresponding ordinary
training statistics. They are empirical thresholds, not formal chronological
SPC control limits.

| Process | T² threshold | Q threshold | Assessment alerts |
|---|---:|---:|---:|
| AlCu | 258.819 | 60.379 | 11/970 (1.13%) |
| WTi | 419.905 | 60.568 | 15/348 (4.31%) |

Each row is placed into one of four descriptive states: normal, T²-only,
Q-only, or both-alarm. This gives an engineer a short review queue. It does not
identify a validated fault.

## 4. AlCu excursion review

Three AlCu rows were selected by a fixed rule: the largest threshold ratio in
each non-normal category. Two are training diagnostics and one is held out.

| Zero-based row | Split role | State | T²/threshold | Q/threshold |
|---:|---|---|---:|---:|
| 1,113 | training diagnostic | T²-only | 14.70 | 0.14 |
| 6 | training diagnostic | Q-only | 0.44 | 3.77 |
| 139 | held-out diagnostic | both-alarm | 180.71 | 1,568.71 |

Q contributions are squared standardized residuals and add to Q. Signed T²
terms also add to T², but correlated channels make them diagnostic rather than
unique causal effects. For index 1,113, the leading T² channels include
`Sensor_53`, `Sensor_56`, and `Sensor_73`. For index 6, the leading Q channels
include `Sensor_13`, `Sensor_97`, and `Sensor_46`. `Sensor_35` and `Sensor_54`
rank prominently for both statistics at index 139. These are channels to inspect
first, not named root causes.

I also compared valid normal and alerted rows in the ordinary AlCu assessment
set. Exact target-profile groups were resampled together in a 2,000-sample
bootstrap.

| Released-scale summary | Alert minus normal mean | Clustered 95% interval |
|---|---:|---:|
| Mean index | -0.00610 | [-0.01355, 0.00071] |
| Reference-profile deviation | +0.00270 | [-0.00111, 0.00643] |

Both intervals include zero. The 11 alerted rows do not establish a clear shift
in either summary. The small alert group also limits precision.

## 5. Direct 17-target virtual metrology

The primary prediction task estimates all 17 released targets directly. Linear
Regression and Ridge are required baselines. The optional Random Forest uses
300 trees, a minimum leaf size of 5, 70% of features per split, and seed 42.

Random Forest is admitted without using assessment rows. On the ordinary AlCu
training folds, seeds 11, 42, and 73 improve mean RMSE over Ridge by 39.0% to
39.2%. That gate controls the primary AlCu result and the configuration later
used for WTi. A separate grouped-training gate improves RMSE by 34.0% to 34.2%
and controls only the grouped AlCu sensitivity result. The two held-out sets do
not participate in each other's model admission.

### Aggregate ordinary-split errors

| Process | Model | MAE | RMSE | Aggregate R² |
|---|---|---:|---:|---:|
| AlCu | Linear Regression | 0.01030 | 0.10607 | -55.257 |
| AlCu | Ridge | 0.01118 | 0.01486 | 0.001 |
| AlCu | Random Forest | 0.00615 | 0.00870 | 0.622 |
| WTi | Linear Regression | 0.01819 | 0.18202 | -176.846 |
| WTi | Ridge | 0.00721 | 0.01458 | -0.172 |
| WTi | Random Forest | 0.00519 | 0.00829 | 0.604 |

The large Linear Regression RMSE values come from rare, high-error predictions.
Negative R² means the fitted model performs worse than the target-wise mean
benchmark on that assessment set. Random Forest gives the clearest and most
stable improvement, which is why it is retained as the final nonlinear model.

### Random Forest error by released target

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

AlCu per-target R² ranges from 0.517 to 0.823. WTi ranges from 0.488 to
0.758. Full-precision values are in the AlCu and WTi per-target CSV files under
[`results/`](../results/README.md).

The grouped sensitivity gives a more conservative AlCu result:

| Process | Ordinary RMSE / R² | Grouped RMSE / R² | RMSE change |
|---|---:|---:|---:|
| AlCu | 0.00870 / 0.622 | 0.00926 / 0.543 | +6.4% |
| WTi | 0.00829 / 0.604 | 0.00767 / 0.690 | -7.5% |

Repeated profiles add some optimism to the ordinary AlCu score, but they do not
explain the full predictive result. The WTi grouped score improves. This is one
alternate split for each process, so it is a sensitivity result rather than a
general guarantee.

For the primary AlCu Random Forest, normal rows have mean row MAE of 0.00611.
The T²-only, Q-only, and both-alarm means are 0.00644, 0.00794, and 0.01263.
The both-alarm group has four rows. The pattern is worth checking during review,
but the subgroup is too small for a reliability threshold.

## 6. WTi replication and zero sensitivity

The WTi scaler, PCA model, empirical thresholds, Ridge penalty, and model fits
use WTi training data. An AlCu-fitted estimator is never applied to WTi. The
script verifies that the Random Forest configuration matches the model admitted
by the primary AlCu training-only gate before fitting it on WTi.

WTi has 333 normal and 14 alerted valid rows in its ordinary assessment set.
The alert-minus-normal mean-index difference is +0.00186 with a clustered 95%
interval of [-0.00420, 0.00768]. The reference-profile deviation difference is
+0.00084 with an interval of [-0.00137, 0.00405]. These intervals also include
zero.

The all-zero sensitivity is strongest when the unknown row falls in assessment:

| Process and split | Zero-row location | RF RMSE excluded | RF RMSE included |
|---|---|---:|---:|
| AlCu ordinary | training | 0.008698 | 0.008715 |
| AlCu grouped | assessment | 0.009258 | 0.020967 |
| WTi ordinary | assessment | 0.008285 | 0.041549 |
| WTi grouped | training | 0.007666 | 0.008124 |

This confirms that the unexplained record can distort Y-based assessment. It
does not explain why the record is zero.

## 7. Limits and conclusion

- The variables are anonymous, so channel rankings cannot be converted into
  equipment diagnoses without process metadata.
- There is no chronology or known-good period. The work is retrospective
  multivariate process-state monitoring, not formal chronological SPC.
- PCA fitting and empirical threshold estimation use the same training rows.
  Alert rates are descriptive and not calibrated false-alarm rates.
- No USL or LSL supports Cp or Cpk. No fault labels support validated fault
  detection.
- The released scales do not support physical AlCu-versus-WTi error ranking.
- WTi is a second public process dataset, not an external production
  qualification or a production-ready controller.

The main result is a reproducible engineering workflow. PCA reduces each
process to a smaller set of variation directions, T² and Q/SPE separate two
types of statistical excursions, and contribution plots identify channels for
review. The direct 17-target Random Forest retains useful released-scale
predictive signal under both ordinary and grouped assessment. The evidence
supports retrospective monitoring and virtual-metrology analysis on this
release. It does not support physical metrology, formal control limits,
validated fault detection, or production deployment.
