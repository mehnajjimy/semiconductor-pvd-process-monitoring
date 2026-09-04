# Independent Audit Prompt for Pre-Audit v0.9

Copy and paste the prompt below into a fresh ChatGPT Work chat with the complete
project folder attached. Do not label the project v1.0 until the independent
review is complete and all required corrections have been made.

---

Act as an independent technical reviewer of my completed semiconductor
process-engineering portfolio project, **Multivariate PVD Process Monitoring
and Virtual Metrology**.

You did not build this project. Treat it as **pre-audit v0.9**. Do not trust the
README, report, code comments, saved CSVs, figures, or conclusions by default.
Independently inspect and rerun the work. The first public release should become
v1.0 only after I address your audit findings.

## Review Rules

- Work read-only against the supplied canonical repository. Do not edit,
  rewrite, delete, rename, commit, or push project files.
- You may use a separate temporary review directory or environment if needed.
- Do not fix issues yet. Diagnose and report them.
- Do not silently fill missing metadata or infer physical meanings.
- If something cannot be verified, state exactly what is missing and classify
  the limitation. Do not guess.
- Do not accept a result merely because a script prints “complete.”
- Recalculate important quantities from the raw CSVs and compare them with the
  saved outputs.
- Distinguish a code defect from a documented dataset limitation.
- Judge beginner-readable code by correctness and explainability, not by how
  abstract or production-oriented it is.

## Canonical Scope

The repository should contain exactly one analysis path:

1. `src/01_data_audit.py`
2. `src/02_preprocess_and_quality.py`
3. `src/03_pca_monitoring.py`
4. `src/04_excursion_analysis.py`
5. `src/05_virtual_metrology.py`
6. `src/06_wti_validation.py`

It should also contain `README.md`, `requirements.txt`, `.gitignore`,
`GITHUB_GUIDE.md`, `report/technical_report.md`,
`report/career_materials.md`, this audit prompt, generated `results/` and
`figures/`, and unchanged source files under `data/raw/`. There should be no
notebook, dashboard, API, package framework, Docker setup, cloud deployment, or
alternate competing implementation.

## Authoritative Dataset Facts to Verify

Use primary sources, not a mirror:

- Official record:
  https://doi.org/10.5281/zenodo.16881338
- Associated paper:
  https://doi.org/10.3233/FAIA251482
- Dataset license: CC BY 4.0.

Verify the attached raw files against the official release:

| File | Expected shape | Expected MD5 |
|---|---:|---|
| `X_pvd_AlCu.csv` | 4,848 × 97 | `9dd4bfc44ffb6a570e519000e07e9fe8` |
| `Y_pvd_AlCu.csv` | 4,848 × 17 | `99b845acd32ae57060e194e407c08f19` |
| `X_pvd_WTi.csv` | 1,740 × 108 | `cfc27ea3bdf780bcdf6dfdc5711836c1` |
| `Y_pvd_WTi.csv` | 1,740 × 17 | `ab6a3b425d18d6f88f0466a7c40b340a` |

The release describes wafer-level instances, aggregated process-sensor
statistics in X, and 17 post-process target measurements in Y. The public files
are already preprocessed and anonymized; they are not raw time-series traces.

The following are unavailable or not independently verifiable and must not be
invented:

- a row ID or X/Y join key;
- chronological row order or timestamps;
- lot, wafer, recipe, tool, chamber, and maintenance identifiers;
- a mapping from `Sensor_n` to a physical sensor or unit;
- target inverse scaling or physical thickness units;
- a usable coordinate mapping for `RAW_VALUE_t_1` through
  `RAW_VALUE_t_17`;
- known-good/fault labels;
- valid USL/LSL values;
- a physical explanation for the all-zero records.

X/Y pairing is positional because row counts match and that is the intended
released supervised-learning structure, but no stored key independently proves
the pairing. Released values lie in [0, 1), but the transformation is
undocumented. Derived outputs must be described as released-scale quantities,
not physical thickness or physical uniformity.

## Data Audit Checks

Independently verify:

- numeric data types, shapes, headers, missing values, infinities, and ranges;
- exact duplicate X rows and columns;
- constant and near-constant features;
- combined X/Y duplicate rows;
- exact duplicate 17-target profiles;
- the absence of any raw-file modification.

Expected structures that must be confirmed, not assumed:

- AlCu has 1,203 later rows that repeat an earlier exact target profile,
  3,645 unique target profiles, and a largest repeated group of 198 rows.
- WTi has 93 later repeated target rows, 1,647 unique target profiles, and a
  largest repeated group of 10 rows.
- There are no exact duplicate combined X/Y rows.
- AlCu zero-based target row 2,586 is all zero.
- WTi zero-based target row 936 is all zero.
- Training-only screening removes AlCu `Sensor_26` and `Sensor_63` at the
  predeclared 98% modal-share rule and retains all 108 WTi features.

Check that raw CSVs are ignored by Git, while `data/raw/README.md` preserves the
official source, license, filenames, and checksums.

## Split and Leakage Audit

The intended design is:

- fixed random state 42;
- ordinary shuffled 80/20 split as the primary validation;
- a separate sensitivity split grouped by the complete exact 17-target
  profile;
- identical target profiles are not called duplicate wafers;
- all-zero rows remain in X-only monitoring, are excluded from primary
  Y-dependent summaries and models, and are included only in explicit
  sensitivity rows.

Independently reconstruct the assignments and verify:

| Process and split | Train | Test | Profile groups in both | Test rows with profile seen in train | Zero row |
|---|---:|---:|---:|---:|---|
| AlCu ordinary | 3,878 | 970 | 214 | 308 | train |
| AlCu grouped sensitivity | 3,768 | 1,080 | 0 | 0 | test |
| WTi ordinary | 1,392 | 348 | 24 | 28 | test |
| WTi grouped sensitivity | 1,393 | 347 | 0 | 0 | train |

Audit every fit boundary:

- feature decisions must not use the relevant assessment outcome;
- scalers, PCA, reference profiles, Ridge tuning, and fitted regressors must use
  training rows only;
- scaling must be fitted inside Ridge cross-validation folds;
- grouped Ridge folds must keep exact profiles together;
- the Random Forest admission gate must use training-only folds and never
  held-out metrics;
- WTi models, scaler, PCA, and thresholds must be fitted on WTi, not transferred
  from an AlCu-fitted estimator;
- only the fixed Random Forest configuration selected by the AlCu training-only
  gate may be refitted on WTi.

Flag any subtle leakage, including preprocessing selected using an alternate
split's assessment rows. If the same feature set is used for both split
designs, independently confirm that training-only screening gives the same
decision under both designs.

## Released-Scale Output Metrics

Recalculate and inspect:

- released-scale mean index;
- released standard deviation, range, and coefficient of variation as
  descriptive released-scale summaries only;
- reference-profile shape deviation.

The shape deviation should use point-specific medians from valid primary
training rows, divide out those point baselines, remove each row's overall
level, and calculate RMS distance from one. It should be undefined for an
all-zero row.

Confirm that no derived quantity is called physical thickness, wafer
uniformity, radial behavior, or center-to-edge behavior. The public point
coordinates and common physical scale needed for those claims are unavailable.

## PCA and Multivariate Monitoring

Independently reproduce:

- training-only standardization;
- full PCA fit;
- smallest retained component count reaching at least 90% cumulative explained
  variance;
- PCA scores and reconstruction;
- Hotelling T² = sum of squared retained scores divided by retained
  eigenvalues;
- Q/SPE = squared norm of the standardized reconstruction residual;
- empirical 99th-percentile thresholds from PCA training rows;
- normal, T²-only, Q-only, and both-alarm categories.

Expected headline values to challenge:

| Process | Retained PCs | Retained variance | T² threshold | Q threshold | Ordinary held-out any-alert rate |
|---|---:|---:|---:|---:|---:|
| AlCu | 34 | 0.905459 | 258.818841 | 60.379356 | 11/970 = 0.011340 |
| WTi | 35 | 0.901574 | 419.905255 | 60.567729 | 15/348 = 0.043103 |

The scaler/PCA fitting rows and empirical-threshold rows are the same. Confirm
the documentation treats these as descriptive historical cutoffs, not
independently calibrated false-alarm limits. There is no known-good baseline or
chronology, so the project must say retrospective multivariate process-state
monitoring, statistical excursions, and empirical alert thresholds—not formal
chronological SPC, validated fault detection, or production control limits.

## Contribution Analysis

Verify that:

- three selected nonzero-output AlCu cases are zero-based indices 1,113
  (training, T²-only), 6 (training, Q-only), and 139 (held-out, both-alarm);
- squared standardized residual contributions sum exactly to each selected Q;
- signed original-variable T² terms sum exactly to each selected T²;
- T² terms are disclosed as diagnostic and nonunique under correlation;
- top anonymous channels are investigation candidates, never physical root
  causes;
- documentation makes clear that only index 139 is a held-out contribution
  example.

## Excursions Versus Released Outputs

Reproduce the any-alert versus normal comparisons on valid ordinary held-out
rows. The 2,000-sample bootstrap must resample exact target-profile groups,
not individual rows independently.

Expected results to verify:

| Process | Metric | Normal n | Alert n | Alert minus normal mean | Clustered 95% interval |
|---|---|---:|---:|---:|---:|
| AlCu | Released mean index | 959 | 11 | -0.006099 | [-0.013548, 0.000715] |
| AlCu | Profile shape deviation | 959 | 11 | 0.002701 | [-0.001109, 0.006427] |
| WTi | Released mean index | 333 | 14 | 0.001863 | [-0.004203, 0.007681] |
| WTi | Profile shape deviation | 333 | 14 | 0.000839 | [-0.001371, 0.004053] |

All intervals include zero. Confirm the project says the comparisons do not
establish a clear released-output difference and acknowledges the small alert
groups. It must not claim proof of no association.

## Virtual Metrology

The primary task is direct native multi-output prediction of all 17 released
targets. Independently verify:

- required Linear Regression and Ridge models;
- training-only Ridge alpha selection over the saved candidate grid;
- aggregate cell-weighted MAE and RMSE;
- uniform-average multi-output R²;
- all 17 per-target MAE, RMSE, and R² values;
- secondary released mean-index and reference-profile-deviation errors;
- ordinary and grouped-profile assessment metrics;
- row-level errors joined to frozen X-only process-state labels;
- with/without all-zero sensitivity.

The optional Random Forest should use exactly:

- 300 trees;
- minimum leaf size 5;
- maximum feature share 0.70;
- random state 42 for the reported model.

Its admission rule is at least 10% mean-RMSE improvement over selected Ridge in
training-only five-fold validation for both AlCu split designs and each seed
11, 42, and 73. Reproduce the six gate rows. Expected improvements are about
39.0–39.2% for the ordinary design and 34.0–34.2% for the grouped design.
Held-out results must not participate in admission.

Expected zero-excluded held-out results to challenge:

| Process | Validation | Model | MAE | RMSE | Aggregate R² |
|---|---|---|---:|---:|---:|
| AlCu | Ordinary | Linear | 0.010305 | 0.106073 | -55.257138 |
| AlCu | Ordinary | Ridge | 0.011183 | 0.014856 | 0.000826 |
| AlCu | Ordinary | Random Forest | 0.006147 | 0.008698 | 0.621850 |
| AlCu | Grouped | Random Forest | 0.006167 | 0.009258 | 0.543233 |
| WTi | Ordinary | Linear | 0.018190 | 0.182019 | -176.846442 |
| WTi | Ordinary | Ridge | 0.007208 | 0.014580 | -0.172465 |
| WTi | Ordinary | Random Forest | 0.005194 | 0.008285 | 0.604213 |
| WTi | Grouped | Random Forest | 0.005126 | 0.007666 | 0.690301 |

Investigate the very poor Linear RMSE/R² rather than assuming a formatting
error. Confirm whether rare high-leverage states create extreme predictions
and whether documentation explains that MAE alone hides them.

AlCu grouped Random Forest RMSE is about 6.4% higher than ordinary and R² falls
from 0.622 to 0.543. WTi grouped performance improves. Confirm the project says
the AlCu repeated-profile structure adds some optimism without claiming the
profiles are duplicate wafers.

## Error by Process State

Recalculate primary Random Forest row MAE by frozen ordinary-split PCA state.
Expected AlCu means are 0.006115 for 959 normal rows, 0.006437 for five
T²-only, 0.007945 for two Q-only, and 0.012627 for four both-alarm rows.

WTi valid primary assessment has 333 normal, two T²-only, seven Q-only, and five
both-alarm rows. The unknown all-zero WTi row is a sixth both-alarm point but is
excluded from primary Y modeling. Confirm all documentation makes the
conditional analysis and very small alarm samples explicit. No reliability
rule or production gating claim is justified.

## All-Zero Sensitivity

Verify both records remain in X-only PCA/process-state outputs and are both
alarms, without assigning meaning:

- AlCu index 2,586: both-alarm; ordinary training; grouped assessment.
- WTi index 936: both-alarm; ordinary assessment; grouped training.

Recalculate the saved sensitivity tables. Key Random Forest changes are:

- AlCu ordinary RMSE 0.008698 excluded versus 0.008715 included;
- AlCu grouped RMSE 0.009258 excluded versus 0.020967 included;
- WTi ordinary RMSE 0.008285 excluded versus 0.041549 included;
- WTi grouped RMSE 0.007666 excluded versus 0.008124 included.

Do not infer missing deposition, a failed wafer, a sentinel, or any other
physical meaning.

## WTi Generalization Framing

Confirm WTi is a separately fitted replication of the workflow, not validation
of an AlCu-fitted model. Cross-process raw RMSE values are on potentially
different undocumented released scales and must not be used as a physical
ranking. The comparison figure should use within-process aggregate R² for
predictive signal and separately report process-specific monitoring behavior.

## Reproducibility Run

Create or use an isolated Python environment with the exact versions in
`requirements.txt`. Run these commands in order from the repository root:

```bash
python src/01_data_audit.py
```

```bash
python src/02_preprocess_and_quality.py
```

```bash
python src/03_pca_monitoring.py
```

```bash
python src/04_excursion_analysis.py
```

```bash
python src/05_virtual_metrology.py
```

```bash
python src/06_wti_validation.py
```

Record every exit code and warning. Do not claim success if any command fails.
Verify regenerated CSVs and all 13 PNG figures are nonempty, readable, and
consistent with their source calculations. Visually inspect every figure for
labels, clipping, misleading axes, stale titles, or unsupported physical
language.

Compare regenerated outputs with the pre-run snapshot. Look for nondeterminism,
hard-coded results, stale artifacts, duplicate obsolete outputs, and generated
files that no script owns.

## Code and Public-Material Review

Check that:

- code uses simple functions and normal pandas/NumPy/scikit-learn operations;
- there are no unnecessary frameworks, classes, command-line systems, or
  competing implementations;
- all explanatory Python comments are lowercase;
- public writing uses normal grammar and restrained engineering language;
- every number in `README.md`, `report/technical_report.md`, and
  `report/career_materials.md` traces to a regenerated result;
- `GITHUB_GUIDE.md` checks Git status, remote URL, and local identity before
  any push;
- raw data are not staged for GitHub;
- no unsupported sensor identity, unit, target geometry, timestamp, root cause,
  capability, chronological SPC, fault-detection, or production-readiness claim
  appears;
- the project is appropriately labeled pre-audit v0.9.

Specifically challenge whether the long but stage-local `05` and `06` scripts
remain understandable to a beginner. Do not require abstraction merely to
reduce line count, but flag duplicated logic that creates a real maintenance or
explanation risk.

## Required Response Format

Return findings in exactly these sections:

### CRITICAL

Issues that invalidate results, introduce material leakage, or materially
misrepresent the project. For each issue, name the exact file, function/output,
and evidence.

### IMPORTANT

Issues that should be corrected before public v1.0. Name the exact location,
why it matters, and the smallest defensible correction.

### MINOR

Improvements worth making that do not undermine the central analysis.

### VERIFIED

List the major components you independently recalculated or reran and found
defensible. Include commands run, exit status, and evidence—not a generic
summary.

### VERDICT

End with exactly one:

- **GREEN:** Safe to publish as currently written.
- **YELLOW:** Fundamentally sound, but fix the listed items before publishing.
- **RED:** Major methodological problems require correction.

Do not return GREEN unless you actually inspected the raw data, source code,
regenerated outputs, checked the key mathematics and leakage boundaries, and
fact-checked the public materials. If the environment prevents a required
check, identify that limitation instead of assuming success.

---
