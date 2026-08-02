# Data 002 Prospective Protocol

**Protocol status:** Frozen prospective design; execution and findings accepted

**Project status:** Findings and public interpretation accepted

**Execution status:** Compatibility, pilot, 540-condition replication grid,
evidence review, and frozen analysis accepted

Authorization commit `0d961d11edc03d3b3994027881431a8e4aec862e`
authorizes only the prospectively fixed 16-condition compatibility replay,
executed sequentially with a 900-second per-condition timeout and a 2 GiB
free-disk gate. The replay is complete and accepted: all 16 newly executed
conditions passed with exact probability and metric agreement and zero reused
conditions. The accepted report SHA-256 is
`9741f7e13b7faec0b27c4ad2404ecfb890c756c605a1f53f1259ca3d124eb19b`.

Wall time was 108.625 seconds, summed condition time was 103.725 seconds, and
the longest condition took 28.487 seconds. Free disk at preflight was
67,696,033,792 bytes.

Compatibility acceptance establishes that the frozen historical evidence can
be reused under the reviewed execution environment. It is not treatment
evidence and does not answer the primary Data 002 comparison. Compatibility
was followed by the separately authorized and accepted metric-blind pilot,
540-condition replication execution, execution-evidence review, frozen
analysis, and scientific review.

This document preserves the intended study design recorded before pilot
execution or outcome inspection. Its prospective gates, estimands, thresholds,
and failure rules remain the historical basis for the accepted study.

## Hypothesis and comparisons

The primary hypothesis is that Gaussian Copula augmentation and deterministic
class-preserving replication differ in predictive utility when final training
size, target counts, split, scarcity subset, model, and seed are matched.

Primary contrast:

`Gaussian ROC-AUC - replication-control ROC-AUC`

Secondary contrasts:

- `replication-control ROC-AUC - real-only ROC-AUC`
- frozen Data 001 `Gaussian ROC-AUC - real-only ROC-AUC`

The study will report effect estimates and uncertainty. It will not use
statistical significance alone as evidence of practical usefulness.

## Experimental units and pairing

The pairing unit is:

`dataset × scarcity level × seed × model`

All three arms must share:

- the train/test indices;
- the retained scarce-real row identities;
- the fitted preprocessing and model specification;
- the held-out real test labels; and
- the evaluation implementation.

Only the training-table restoration mechanism differs.

## Replication control

For each retained scarce-real training subset:

1. retain every scarce-real row;
2. calculate required additional class counts using Data 001's frozen
   `allocate_class_counts` rule with `minimum_per_class=0`;
3. rank retained rows within each class by the SHA-256 digest of canonical
   UTF-8 JSON binding namespace `data002.replication-order.v1`, dataset,
   retained-fraction token, split seed, class label, and original source-row
   index;
4. break a digest collision by ascending original source-row index;
5. cycle through that outcome-independent ranked order until the required
   additional count is reached;
6. restrict class labels to the exact integer set `{0, 1}`;
7. retain all originals first in accepted reconstruction order, then append
   the class-0 duplicate block followed by the class-1 duplicate block, with
   each block preserving its SHA-256-ranked cycle order; and
8. verify exact total rows, target counts, source-row membership, and repetition
   balance.

Each source row within a class must receive either `floor(k/n)` or `ceil(k/n)`
of the required duplicate assignments, subject to the deterministic remainder
allocation. Canonical JSON uses sorted keys, separators `(",", ":")`,
`ensure_ascii=false`, and `allow_nan=false`. This ordering does not use a
library PRNG.

Model, preprocessing, split, scarcity subset, and prediction behavior must
reuse the compatibility-accepted `src/data002/reconstruction.py` without
modification. Its metric behavior may be reused only during separately
authorized analysis.

The expanded tables will not be persisted as primary artifacts. The source
subset identity, allocation plan, hashes, counts, prediction artifacts,
warnings, runtime, and terminal status will be retained. Replication execution
must not compute or retain predictive metrics.

## Compatibility gate

The completed compatibility gate used this prospectively fixed procedure:

1. reconstruct the selected Data 001 scarce-real baseline conditions from the
   frozen datasets, splits, subsets, preprocessing, and model specifications;
2. fit them in the intended Data 002 environment;
3. compare test indices, labels, predicted probabilities, and recomputed
   metrics with the archived Data 001 baseline predictions; and
4. publish a compatibility report without creating or inspecting any Data 002
   replication treatment.

The replay matrix is fixed at 16 conditions:

| Dataset | Retained fractions | Seeds | Models |
|---|---|---|---|
| Diabetes | 1%, 50% | 0, 29 | Logistic Regression, Random Forest |
| Cleveland | 5%, 50% | 0, 29 | Logistic Regression, Random Forest |

These are the least- and most-scarce conditions shared with the Gaussian arm
and the endpoint seeds. They were selected by design coverage, not by observed
predictive performance.

The reference evidence is Data 001 Gaussian v1.0 at commit
`ecc2b222eca86c47acdf12efd3b8f779b6a29ef9`, including:

- Diabetes SHA-256:
  `19f367e3e3350768f0c144c5d73ee5b355f67a57eaaa86ca7bd8aec594d8b1d0`;
- Cleveland SHA-256:
  `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`;
- NumPy 2.5.1;
- pandas 2.3.3;
- scikit-learn 1.9.0; and
- the model and preprocessing parameters stored in Data 001's frozen
  `baseline_protocol_v1.json`.

The 16 selected reference archives are snapshotted under
`evidence/data001_baseline_replay_v1/`. Their source hashes and selection rule
are bound by `manifest.json`, whose initial SHA-256 is
`caa360e5d0daf34ce1d28633978a2dcc44d7ef44ac0291a9c2d241bd375c6a0e`.
The manifest and every declared archive hash must verify before replay.

Every replay condition must satisfy all of the following:

- exact equality of ordered test indices;
- exact equality of test labels;
- probability shape and `float32` dtype identical to the archive contract;
- finite probabilities within `[0, 1]`;
- elementwise probability agreement with `rtol = 0` and `atol = 1e-6`;
- exact equality of thresholded predictions at the locked `0.5` threshold;
- absolute ROC-AUC and average-precision differences no greater than `1e-12`;
  and
- exact equality of precision, recall, F1, and accuracy.

All 16 conditions must pass. Archive byte equality is not required because
compressed-container metadata is not scientific content. The compatibility
report must show condition-level maxima and failures rather than only a single
pass/fail flag.

Failure does not authorize tolerance relaxation after results are seen. It
requires either:

- a narrower cross-execution interpretation; or
- a prospective new design with contemporaneously executed comparison arms.

## Frozen metric-blind operational pilot

The separately authorized and accepted pilot contained exactly 12 conditions:

- Diabetes at retained fractions 0.01 and 0.50;
- Cleveland at retained fraction 0.05;
- seeds 0 and 29; and
- Logistic Regression and Random Forest.

The metric-blind operational review was limited to:

- runtime, memory, disk, and warnings;
- row and target counts;
- deterministic reconstruction and hashes;
- prediction schema, range, and finiteness booleans, artifact hashes, and
  expected-key identity;
- atomic persistence and resume behavior; and
- failure accounting.

Replication execution and the pilot must not compute, store, print, summarize,
or join predictive metric values. Operational validation may return only
schema/range/finiteness booleans, hashes, counts, warnings, runtime, memory,
disk, and terminal status; it must not return probability values or metrics.
Prediction NPZ files may contain only the locked test indices, labels, and
probabilities. The pilot must not load or join the upstream metric tables.

Pilot prediction artifacts were treated as sealed outcome evidence and were
not interpreted during operational review. All 12 joined the final grid only
after protocol, implementation, environment, and condition contracts were
verified as byte-identical. The pilot completed and was accepted.

## Frozen analysis plan

The primary contrast is Gaussian ROC-AUC minus replication ROC-AUC in each of
18 prespecified dataset × retained-fraction × model strata. Seed is the pairing
unit. Each stratum uses 10,000 paired bootstrap resamples of its 30 seed-level
differences, a 95% percentile interval, and NumPy's linear quantiles.

The deterministic analysis seed is derived separately for each stratum from
canonical UTF-8 JSON binding namespace `data002.paired-bootstrap-seed.v1`,
dataset, retained-fraction token, model, contrast, and resample count. The first
eight SHA-256 digest bytes are interpreted as an unsigned big-endian 64-bit
integer for NumPy PCG64.

The practical ROC-AUC threshold is `0.01`. The primary contrast is helpful only
when the entire interval exceeds `+0.01`, harmful only when the entire interval
is below `-0.01`, and otherwise has no clear practically meaningful
difference. No p-values or multiplicity adjustment will be used. Aggregate
counts and heterogeneity views are descriptive.

Analysis was blocked unless all 540 expected replication keys were successful
and checkpoint/prediction reconciliation was exact. Replication metrics could
not exist before this gate passed. After the gate passed, the frozen analysis
was separately authorized and executed exactly once. It reconciled 540
replication keys, 540 Gaussian keys, 660 baseline-source keys, 540 matched
baseline keys, 18 strata, and 30 paired seeds per stratum without silent
dropping or complete-case analysis.

The accepted primary finding is that Gaussian minus replication was negative
in all 18 strata, practically harmful in 10, no-clear in 8, and helpful in
none. Replication minus real-only was harmful in 7 strata. The scientific
findings and public README/notebook interpretation are accepted.

## Resource envelope

The replication grid is expected to require 540 model fits and roughly
90–110 MB of prediction artifacts. Historical Data 001 timings suggest about
three hours of model fitting on the reference host, with Random Forest fits
dominating runtime.

The proposed design envelope is sequential execution, a 900-second
per-condition timeout, a 2 GiB free-disk gate, atomic condition persistence,
verified resume, and exclusive run locking. These values are design decisions,
not execution authorization.

## Interpretation limits

The study supports conclusions only for:

- the two anchor datasets;
- the declared scarcity regimes;
- the two predictive models;
- deterministic class-preserving replication as defined here; and
- frozen Gaussian Copula v1 under Data 001's restoration design.

It cannot establish that synthetic data generally helps or fails, that
replication is an optimal control, or that either method adds genuinely new
information.
