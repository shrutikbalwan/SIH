# V1 vs V2 False-Positive Analysis

## Overview

- **V1 model**: `ira_cnn_best.keras`
- **V2 model**: `ira_cnn_v2_best_loss.keras`
- **Test set**: 772 held-out negatives from `split_manifest.csv`
- **Threshold evaluated**: 0.50

## Section 1 - FPR Change

| | V1 | V2 |
|---|---|---|
| Test negatives | 772 | 772 |
| FP @ 0.50 | 3 | 112 |
| FPR @ 0.50 | 0.39% | 14.51% |

## Section 2 - Test Set Verification

Files loaded from `split_manifest.csv` (split=test, label=0).
Cross-checked against `dataset/splits/test/negative/` (V1 physical directory).

Category breakdown:
| Category | Count |
|---|---|
| ambient | 200 |
| speech | 572 |

## Section 3 - Transition Matrix (V1?V2 @ 0.50)

| Transition | Count |
|---|---|
| TN?TN (correct both) | 658 |
| **TN?FP (V2 regression)** | **111** |
| FP?TN (V2 improvement) | 2 |
| FP?FP (wrong both) | 1 |

> [!IMPORTANT]
> **111** negatives were correctly rejected by V1 but falsely triggered by V2.

## Section 4 - Per-Category FP Breakdown

### V1
| Category | N | FP | FPR | Median |
|---|---|---|---|---|
| speech | 572 | 3 | 0.52% | 0.0000 |
| ambient | 200 | 0 | 0.00% | 0.0000 |
| other | 0 | 0 | N/A | N/A |

### V2
| Category | N | FP | FPR | Median |
|---|---|---|---|---|
| speech | 572 | 112 | 19.58% | 0.1065 |
| ambient | 200 | 0 | 0.00% | 0.0002 |
| other | 0 | 0 | N/A | N/A |

### TN?FP Regressions by Category
| Category | Count | % of regressions |
|---|---|---|
| speech | 111 | 100.0% |

> **Concentration**: speech negatives account for **100.0%** of the TN?FP regressions.

## Section 5 - Score Distribution (all 772 test negatives)

| Stat | V1 | V2 | V2-V1 |
|---|---|---|---|
| mean | 0.0075 | 0.1821 | 0.1747 |
| median | 0.0000 | 0.0339 | 0.0292 |
| p90 | 0.0035 | 0.6901 | 0.6830 |
| p95 | 0.0135 | 0.8735 | 0.8636 |
| p99 | 0.1801 | 0.9809 | 0.9772 |
| max | 0.8903 | 0.9989 | 0.9985 |

## Section 6 - Hard-Negative Cross-Check

- Training hard negatives found: 0
- Exact file overlap with V2 test FPs: **0**
- Result: ? No exact file leakage

## Section 7 - Top 30 Regressions

| Rank | Category | V1 | V2 | Delta | Filename |
|---|---|---|---|---|---|
| 1 | speech | 0.0002 | 0.9988 | 0.9985 | 1737_1737-146161-0005_neg_010.wav |
| 2 | speech | 0.0009 | 0.9989 | 0.9981 | 1737_1737-146161-0003_neg_006.wav |
| 3 | speech | 0.0011 | 0.9939 | 0.9927 | 1737_1737-146161-0012_neg_005.wav |
| 4 | speech | 0.0000 | 0.9860 | 0.9860 | 1737_1737-146161-0004_neg_011.wav |
| 5 | speech | 0.0002 | 0.9844 | 0.9842 | 1737_1737-146161-0016_neg_008.wav |
| 6 | speech | 0.0000 | 0.9812 | 0.9812 | 1737_1737-146161-0001_neg_012.wav |
| 7 | speech | 0.0000 | 0.9808 | 0.9808 | 1737_1737-146161-0011_neg_003.wav |
| 8 | speech | 0.0000 | 0.9789 | 0.9789 | 1737_1737-146161-0003_neg_000.wav |
| 9 | speech | 0.0127 | 0.9892 | 0.9765 | 1088_1088-134318-0003_neg_012.wav |
| 10 | speech | 0.0035 | 0.9749 | 0.9714 | 1737_1737-146161-0012_neg_001.wav |
| 11 | speech | 0.0002 | 0.9715 | 0.9714 | 1737_1737-146161-0001_neg_010.wav |
| 12 | speech | 0.0000 | 0.9677 | 0.9677 | 1737_1737-146161-0008_neg_005.wav |
| 13 | speech | 0.0003 | 0.9672 | 0.9669 | 5789_5789-70653-0011_neg_006.wav |
| 14 | speech | 0.0052 | 0.9713 | 0.9660 | 6848_6848-252323-0025_neg_002.wav |
| 15 | speech | 0.0000 | 0.9627 | 0.9627 | 1737_1737-146161-0003_neg_003.wav |
| 16 | speech | 0.0002 | 0.9614 | 0.9612 | 1737_1737-146161-0002_neg_002.wav |
| 17 | speech | 0.0020 | 0.9552 | 0.9532 | 5789_5789-70653-0012_neg_000.wav |
| 18 | speech | 0.0005 | 0.9516 | 0.9511 | 5789_5789-70653-0035_neg_004.wav |
| 19 | speech | 0.0062 | 0.9511 | 0.9449 | 1737_1737-146161-0001_neg_000.wav |
| 20 | speech | 0.0000 | 0.9434 | 0.9434 | 1737_1737-146161-0016_neg_001.wav |
| 21 | speech | 0.0000 | 0.9426 | 0.9426 | 1737_1737-146161-0000_neg_007.wav |
| 22 | speech | 0.0000 | 0.9426 | 0.9425 | 5789_5789-70653-0023_neg_003.wav |
| 23 | speech | 0.0033 | 0.9317 | 0.9284 | 5789_5789-70653-0021_neg_006.wav |
| 24 | speech | 0.0001 | 0.9229 | 0.9228 | 1737_1737-146161-0001_neg_015.wav |
| 25 | speech | 0.0000 | 0.9173 | 0.9173 | 1737_1737-146161-0002_neg_011.wav |
| 26 | speech | 0.0001 | 0.9129 | 0.9128 | 5789_5789-70653-0025_neg_000.wav |
| 27 | speech | 0.0000 | 0.9127 | 0.9127 | 5789_5789-70653-0000_neg_010.wav |
| 28 | speech | 0.0000 | 0.9095 | 0.9095 | 1737_1737-146161-0002_neg_005.wav |
| 29 | speech | 0.0000 | 0.9063 | 0.9063 | 5789_5789-70653-0028_neg_006.wav |
| 30 | speech | 0.0056 | 0.9003 | 0.8947 | 5789_5789-70653-0017_neg_004.wav |

Files copied to: `v2_false_positive_analysis/top_regressions/`

## Section 8 - Threshold Feasibility (Diagnostic Only)

Any threshold in [0.50, 0.99] where REAL TPR=95%, ALL TPR=95%, FPR=1%?
**No. No such threshold exists in this range.**

> [!NOTE]
> Diagnostic only. No production threshold selected.

## Conclusion

The FPR regression (0.39% ? 14.51%) is **concentrated in speech negatives** (100.0% of regressions).

The test negative file list is **identical** between V1 and V2 evaluations - the regression is a genuine model behaviour difference, not a dataset difference.

Do NOT retrain. Do NOT quantize. Do NOT run streaming evaluation yet.
