# Data 002 — Synthetic Restoration Controls

**Status:** Analysis complete; scientific findings accepted

Under the frozen two-dataset, two-model scarcity protocol, Gaussian Copula
augmentation added no predictive value beyond deterministic class-preserving
replication. It was practically harmful in 10 of 18 strata and never helpful.

Replication itself was harmful in 7 strata, indicating that restoration
without new information explains part, but not all, of the degradation.

## Public result

Data 002 asks whether Gaussian synthesis adds predictive value beyond a
no-new-information restoration control when final training size and target
class counts are matched. The prespecified primary comparison is:

`ROC-AUC(Gaussian augmentation) − ROC-AUC(deterministic replication)`

“Practically harmful” means that the entire 95% paired-bootstrap interval was
below the prespecified `−0.01` ROC-AUC threshold. “Helpful” requires the
complete interval to exceed `+0.01`; all other intervals are labeled “no clear
practically meaningful difference.”

Across the 18 primary strata, all 18 point estimates were negative and all 18
intervals were entirely below zero. Ten strata met the prespecified practical-
harm threshold, eight showed no clear practically meaningful difference, and
none were helpful.

The secondary frozen comparisons were:

| Contrast | Harmful | No clear practically meaningful difference | Helpful |
|---|---:|---:|---:|
| Replication minus scarce real-only | 7 | 11 | 0 |
| Gaussian minus scarce real-only | 12 | 6 | 0 |

See the executed [results notebook](notebooks/01_replication_control_results.ipynb)
for the accessible forest plot, all interpretation counts, and the exact
18-stratum primary-result table.

## Study design

The three matched training conditions were:

1. scarce real data only;
2. scarce real data restored by deterministic, class-preserving replication;
3. scarce real data augmented with the frozen Gaussian Copula v1 method from
   Data 001.

The frozen grid covered Diabetes and Cleveland Heart Disease, Logistic
Regression and Random Forest, the prespecified scarcity fractions, and 30
paired seeds per stratum. Intervals used 10,000 paired seed-level bootstrap
resamples with the frozen deterministic procedure. No p-values or multiplicity
adjustment were used.

## Accepted evidence

- [Analysis report](results/analysis/replication_v1/analysis_report.json)
- [Scientific review record](results/provenance/replication_analysis_review_v1.json)

The notebook verifies the accepted stratum-summary SHA-256 before loading and
loads no prediction arrays, seed-level metric document, or upstream metric
table.

## Public release boundary

This release contains the scientific configurations and implementation, the
portable Data 001 metric bundle, the accepted analysis outputs, and the
portable tests.

Host-specific launch records, environment inventories, logs, checkpoints,
prediction archives, operational tests, and internal coordination documents
are intentionally excluded. Run `python -m pytest -q` to validate the 98-test
portable boundary.

## Limitations

- Two binary-classification datasets
- Two model families
- ROC-AUC only
- Frozen scarcity and restoration protocol
- Replication is a matched control, not a statistically neutral intervention
- No causal or universal synthetic-data claim

Data 002 is the focused control study following Data 001. Its conclusion is
limited to the frozen protocol above.
