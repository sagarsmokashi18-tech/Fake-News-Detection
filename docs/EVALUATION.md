# Model Evaluation and Leakage Review

## Dataset inspection

The dataset combines `True.csv` (21,417 REAL rows) and `Fake.csv` (23,481 FAKE rows), 44,898 rows total.

| Check | Observed result |
|---|---:|
| Empty normalized article text | 631 |
| Repeated normalized text rows before filtering | 6,280 (includes empty rows) |
| Rows after empty removal and exact normalized-text deduplication | 38,617 |
| Remaining REAL / FAKE rows | 21,170 / 17,447 |
| Cross-label exact normalized non-empty text groups | 0 |
| Repeated titles in raw data | 6,178 repeated rows |

The training process removes empty text and deduplicates exact normalized article text **before** splitting, so those exact repeats cannot appear on both sides of its random holdout. The title-group holdout keeps identical normalized titles together. A title grouping is only a partial proxy for repeated/near-duplicate reporting; the source files do not provide dependable publisher identities, and a publisher-disjoint evaluation was not possible.

## Leakage and representativeness risks

- **Source and topic confounding:** `True.csv` contains `politicsNews` and `worldnews`; `Fake.csv` contains `Government News`, `Middle-east`, `News`, `US_News`, `left-news`, and `politics`. These subject values are disjoint by label. The subject field is not supplied to the model, but article phrasing, wire-service style, dates, and source conventions can still identify the originating corpus.
- **Temporal confounding:** REAL dates range from January 2016 through December 2017; FAKE dates range from March 2015 through February 2018. A chronological holdout consequently has a highly shifted class mix.
- **Near-duplicate limitation:** normalized exact article copies were removed. Exact repeated titles were isolated in the title-group split; similar but non-identical story bodies and hidden publisher duplication were not comprehensively deduplicated.
- **Evaluation domain:** every split comes from the same historical dataset. There are no independent publishers, current articles, or independent human fact-check labels.
- **Calibration:** no probability calibration was performed. The UI therefore does not display the classifier's `predict_proba` values as confidence.

## Metrics

All metrics below were obtained by fitting the same Logistic Regression + TF-IDF configuration on the relevant training partition. Confusion matrices use row/column order **FAKE, REAL**.

### Deduplicated stratified random holdout

This reproduces the prior 98.76% result: normalized empty and exact duplicate text rows are removed before an 80/20 stratified split (`random_state=42`).

- Train/test size: 30,893 / 7,724
- Test support: FAKE 3,490; REAL 4,234
- Accuracy: **0.9876**
- Macro precision / recall / F1: **0.9878 / 0.9871 / 0.9874**
- Confusion matrix: `[[3429, 61], [35, 4199]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| FAKE | 0.9899 | 0.9825 | 0.9862 | 3,490 |
| REAL | 0.9857 | 0.9917 | 0.9887 | 4,234 |

This is the same internal 98.76% headline result from the earlier phase, not an independent or real-world validation.

### Normalized-title group holdout

`GroupShuffleSplit(test_size=0.2, random_state=42)` assigns all rows with the same normalized title to one side. It is a check for title reuse, not publisher generalization.

- Train/test size: 30,905 / 7,712
- Test support: FAKE 3,571; REAL 4,141
- Accuracy: **0.9904**
- Macro precision / recall / F1: **0.9906 / 0.9901 / 0.9903**
- Confusion matrix: `[[3522, 49], [25, 4116]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| FAKE | 0.9930 | 0.9863 | 0.9896 | 3,571 |
| REAL | 0.9882 | 0.9940 | 0.9911 | 4,141 |

This does not remove all possible near-duplicate stories or source-style leakage. The dataset itself is an unusually separable source mixture.

### Chronological holdout (diagnostic only)

The separate `date` and `dates` columns were coalesced and parsed (38,611 of 38,617 rows had parseable dates). Training uses dates on or before the 80th-percentile cutoff **2017-10-18**; test uses later dates. No class balancing was imposed on this chronological test.

- Train/test size: 30,928 / 7,683
- Test support: FAKE 895; REAL 6,788 (strongly shifted from the training distribution)
- Accuracy: **0.9928**
- Macro precision / recall / F1: **0.9780 / 0.9877 / 0.9828**
- Confusion matrix: `[[878, 17], [38, 6750]]`

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| FAKE | 0.9585 | 0.9810 | 0.9696 | 895 |
| REAL | 0.9975 | 0.9944 | 0.9959 | 6,788 |

This is a temporal extrapolation within the same dataset, not prospective fact-check validation. The changed label prevalence makes accuracy especially easy to misinterpret.

## Conclusion

The corrected code reproduces high holdout scores after removing exact normalized duplicates before splitting. Grouped and chronological splits also score highly, but neither addresses the dataset's source/subject confounding or independent publisher generalization. The numerical scores must not be marketed as real-world fake-news detection accuracy or evidence that an individual claim is true.
