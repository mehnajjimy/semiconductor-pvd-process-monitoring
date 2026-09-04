# Career Materials

## Project Title

**Multivariate PVD Process Monitoring and Virtual Metrology**

## Resume Bullets

- Built a retrospective multivariate monitoring workflow for 4,848 AlCu and
  1,740 WTi PVD observations, reducing 95–108 anonymized process channels to
  34–35 PCA components and calculating explicit Hotelling T² and Q/SPE
  statistics with training-derived empirical alert thresholds.
- Connected process-state monitoring to engineering investigation by
  reconciling T²/Q channel contributions, comparing held-out alert categories
  with released-scale output summaries, and documenting why anonymous channels
  are review candidates rather than confirmed root causes.
- Predicted 17 released target measurements directly with multi-output Linear,
  Ridge, and a stability-gated Random Forest; achieved ordinary-split aggregate
  R² of 0.622 for AlCu and 0.604 for separately fitted WTi, then checked
  identical-target-profile splits and all-zero-record sensitivity.

## 15-Second Explanation

I built a semiconductor PVD process-monitoring project that compresses about
100 correlated measurements with PCA, uses T² and Q to flag unusual process
states, shows which anonymous channels deserve investigation, and tests
released-scale virtual metrology on separate AlCu and WTi workflows.

## 30-Second Recruiter Explanation

I used public Infineon PVD data to build a process-engineering project rather
than a leaderboard model. I audited the data, reduced correlated measurements
with PCA, calculated Hotelling T² and Q/SPE for retrospective excursion
monitoring, and connected those states to 17 released target measurements. A
carefully gated Random Forest reached aggregate R² of 0.622 on AlCu and 0.604
on separately fitted WTi. I also checked leakage risk from repeated targets
and clearly limited the claims because units, chronology, specifications, and
fault labels are unavailable.

## 60-Second Technical Explanation

The release contains 4,848 AlCu rows with 97 process inputs and 1,740 WTi rows
with 108 inputs, plus 17 released target measurements per row. I verified the
official checksums, found no missing data, removed only two near-constant AlCu
channels, and used an ordinary fixed split as primary validation. PCA retained
34 AlCu and 35 WTi components for at least 90% variance. I explicitly computed
T² as retained-score distance and Q as squared reconstruction residual, with
99th-percentile empirical thresholds fitted on training rows. Contribution
terms prioritized channels for review without claiming causation.

For virtual metrology, I predicted all 17 targets directly. Linear Regression
showed unstable extreme errors and training-only Ridge tuning selected strong
regularization. One Random Forest entered only after improving Ridge RMSE by
34–39% in training-only folds across both AlCu split designs and three seeds.
Its ordinary-split
aggregate R² was 0.622 for AlCu and 0.604 for an independently fitted WTi
workflow. Grouping identical target profiles reduced AlCu R² to 0.543 but did
not remove the predictive signal. Since the data lack physical units,
timestamps, specifications, and fault labels, I present statistical excursions
and released-scale metrics rather than physical metrology or production SPC.

## Likely Interview Questions and Answers

### Why use PCA?

The inputs are numerous and correlated. PCA converts them into fewer
orthogonal directions that retain most process-input variation, making a
multivariate state easier to monitor. It also provides a reconstruction that
separates modeled variation from residual behavior.

### What is the difference between T² and Q/SPE?

T² measures how far an observation lies from the center within the retained
PCA space. Q measures how much standardized input variation the retained PCA
space cannot reconstruct. A point can be unusual in one sense, both, or
neither, so I keep four alert categories.

### Why are the thresholds called empirical alerts instead of control limits?

They are 99th percentiles from the same training subset used to fit PCA, and
the release has no chronological sequence or documented known-good baseline.
That supports retrospective ranking of unusual states, not an independently
calibrated false-alarm rate or a claim of Phase I/Phase II SPC control limits.

### How did you prevent leakage?

The primary test rows were fixed before model fitting. Feature screening,
scaling, PCA, alert thresholds, reference profiles, and Ridge tuning used
training data only. Ridge scaling was fitted inside each cross-validation
fold. I also kept exactly repeated target profiles together in a second
sensitivity split and reported the change.

### Why was the ordinary split primary if target profiles repeat?

Different X rows with the same Y profile are not proven duplicate observations.
Automatically grouping the primary split would impose an unsupported wafer
identity assumption. I used a standard fixed split for the main estimate and a
grouped split to reveal how sensitive results are to repeated target values.

### Why not remove correlated features before PCA?

Representing correlated variation is the purpose of PCA. I removed only
constant, duplicate, or extremely modal channels that were clearly unusable,
then allowed PCA to summarize the remaining correlation structure.

### Why did Ridge use such strong regularization?

Some statistical excursions become extreme standardized extrapolations when
held out inside cross-validation. Weakly regularized linear coefficients
produced rare but very large errors. Training-only RMSE therefore favored
strong shrinkage. The result is honest evidence that a simple linear mapping
is not stable across all released process states.

### Why was Random Forest allowed?

I defined a narrow gate instead of trying many models: one fixed configuration
had to improve Ridge aggregate RMSE by at least 10% in training-only folds for
both AlCu split designs across three seeds. It improved by 34–39% every time,
so the added nonlinear model was clear, stable, and still explainable. The
held-out rows were not used for that admission decision.

### Does an alert prove bad film quality?

No. In the AlCu held-out data, alert-minus-normal differences in the
released-scale mean index and reference-profile shape deviation both had
clustered bootstrap intervals containing zero. The process state is unusual,
but the public data do not establish a clear released-output penalty.

### Does a high contribution identify root cause?

No. It identifies a measurement channel that contributes strongly to the
statistical excursion and should be investigated first. Correlation,
anonymization, and absent equipment context prevent causal attribution.

### What did you learn from model error by process state?

AlCu both-alarm observations had mean row MAE about twice the normal mean, and
WTi Q-only and both-alarm means were also higher than normal. The groups are
small, so I treat this as a descriptive applicability warning, not a validated
rule.

### How did you handle the all-zero target records?

I retained them in X-based process-state monitoring, excluded them from primary
Y-based summaries and models, and ran with/without sensitivity checks. I did
not label them as missing wafers, faults, or any other physical condition
because the documentation does not provide that meaning.

### Why did you not calculate Cp or Cpk?

The release provides no applicable USL or LSL. Inventing specification limits
would make capability indices meaningless.

### What would you need before production use?

I would need physical units and inverse scaling, point geometry, timestamps,
tool and recipe context, known-good baselines, engineering specifications,
fault and maintenance labels, a leakage-safe chronological validation, and a
controlled prospective evaluation.

## Employer-Specific Emphasis

These are emphasis choices for interviews, not claims about any employer's
specific internal methods.

### Micron

Emphasize high-volume manufacturing discipline: reducing many correlated
signals to an investigation queue, separating statistical excursions from
confirmed defects, and requiring specifications and chronology before formal
control or capability claims.

### onsemi

Emphasize practical process ownership: conservative preprocessing,
reproducible checks, clear escalation from alert to channel review, and honest
communication when released outputs do not show a statistically clear change.

### GlobalFoundries

Emphasize the separately fitted AlCu and WTi workflows. The method transfers
across two processes, while thresholds and model behavior remain
process-specific—an important foundry lesson when products and process modules
differ.

### Intel

Emphasize multivariate process control, metrology reduction, leakage-safe
validation, and applicability-domain thinking. Explain why unusual process
states can deserve human review even when a predictive model still returns a
number.

### Applied Materials

Emphasize equipment/process diagnostics: T² and Q separate unusual modeled
variation from unusual residual correlation, and contribution plots narrow an
alarm from roughly 100 anonymous channels to a short engineering review list.

### Lam Research

Emphasize deposition-process reasoning, excursion triage, and the boundary
between statistical evidence and equipment root cause. Discuss what chamber,
recipe, maintenance, and trace-level metadata would be needed for the next
investigation.

### KLA

Emphasize the connection between process signals, released metrology targets,
and model reliability. Highlight direct multi-output prediction, per-target
errors, and the need to monitor model confidence outside represented process
states.

### Tokyo Electron (TEL)

Emphasize readable engineering implementation, cross-process validation, and
how empirical monitoring could support equipment/process integration once
tool context, sequence, and physical measurements are available.

## Claims to Avoid

- Do not call the released values physical thickness or uniformity.
- Do not call empirical thresholds formal chronological control limits.
- Do not call alerts faults or contribution leaders root causes.
- Do not imply the WTi model is the AlCu model transferred across processes.
- Do not claim production readiness or capability without prospective data and
  specifications.
