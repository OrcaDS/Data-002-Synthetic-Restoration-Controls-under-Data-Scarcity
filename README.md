# Data 002 — Synthetic Restoration Controls

**Does synthetic data add value beyond simply duplicating scarce real rows?**

**Status:** Analysis complete; scientific findings accepted

> Data 002 is a 540-condition matched control study testing whether Gaussian
> Copula augmentation outperforms deterministic row replication when training
> size, class balance, data splits, models, and test sets are held constant.
>
> Across all 18 dataset × scarcity × model settings, the estimated
> Gaussian-versus-replication difference was negative. Gaussian augmentation
> met the prespecified practical-harm threshold in 10 settings, showed no clear
> practically meaningful difference in 8, and was never helpful. Replication
> itself was harmful in 7 settings—showing that restoring table size explains
> part, but not all, of the degradation.

## Explore the replication-control results

- [Executed results notebook](notebooks/01_replication_control_results.ipynb)
  — accessible forest plot, interpretation counts, and the complete
  18-stratum table
- [Stratum-level results](results/analysis/replication_v1/stratum_summaries.json)
  — accepted estimates, intervals, and interpretations
- [Analysis report](results/analysis/replication_v1/analysis_report.json)
  — reconciliation and output accounting
- [Scientific review record](results/provenance/replication_analysis_review_v1.json)
  — accepted conclusion and limitations

The notebook verifies the accepted stratum-summary SHA-256 before loading and
loads no prediction arrays, seed-level metric document, or upstream metric
table.

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

The executed results notebook above provides the accessible forest plot, all
interpretation counts, and the complete 18-stratum primary-result table.

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

## What this project demonstrates

- Matched-control experimental design that isolates the restoration mechanism
- Paired uncertainty estimation across repeated train/test splits
- Deterministic, auditable reconstruction of the replication control
- Prospective practical thresholds and explicit alternative-explanation
  testing
- Reproducible evidence with a 98-test portable validation boundary

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
