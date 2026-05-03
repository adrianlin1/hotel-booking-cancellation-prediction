# Hotel Booking Cancellation Prediction

A machine learning pipeline that predicts whether a hotel reservation will be cancelled, built to support revenue management decisions like overbooking strategy and customer outreach. The deployed model achieves a ~0.70 cross-validated F1 score on a real-world reservation dataset of 119,390 bookings, with a tuned threshold and an asymmetric-cost framing that ties model selection back to operator economics.

**Course:** QST BA 576 — Machine Learning for Business Analytics, Boston University Questrom School of Business (Spring 2026)
**Role:** Team Lead — sole technical implementer (notebook structure, data cleaning, feature engineering, EDA, modeling, hyperparameter tuning, evaluation, and final presentation)

---

## The Question

Hotel revenue management lives or dies by the cancellation forecast. A missed cancellation means an empty room and a lost room-night; a false alarm just triggers a low-cost confirmation outreach call. The cost ratio is not 1:1, so an F1-maximizing model is not automatically the right answer. The right answer is whichever model best matches the operator's asymmetric cost structure — and that reframing changed which model went into deployment.

## Dataset

- 119,390 booking records, 32 raw columns (20 numerical, 12 categorical)
- Target: `is_canceled` (binary)
- After cleaning: 86,631 records, 31 features after engineering
- Class balance after cleaning: 27.5% positive (cancelled), 72.5% negative
- Source: Antonio, Almeida & Nunes (2019), *Hotel booking demand datasets* — publicly available

## Approach

### Feature engineering

- Cyclical encoding for arrival month (sin/cos)
- Country grouping into top-five origins (Portugal, GB, France, Spain, Germany) + Other
- Behavioral aggregates: `total_stay`, `total_guests`, `is_family`
- Investigated and dropped `deposit_type`. The variable looked dominant in raw data (~95% cancellation rate for non-refundable bookings) but only ~1% of records carried that strong signal; the no-deposit and refundable subsets sat near the baseline cancellation rate. Dropping it cost roughly 0.003 F1 and produced a cleaner, more defensible model that doesn't lean on a policy artifact instead of behavioral signal.

### Models compared (eight families, with variants)

- Logistic Regression — baseline, ridge (L2), lasso (L1), forward / backward selection, polynomial features
- KNN
- Decision Tree
- Random Forest
- Bagging
- Gradient Boosting (with HistGradientBoosting used as a runtime workaround on tuning loops)
- AdaBoost
- SVM

### Tuning

- GridSearchCV for fast estimators
- RandomizedSearchCV for slow estimators (SVM, full Gradient Boosting) to control runtime without sacrificing search quality
- `class_weight=balanced` where supported
- Evaluation metrics: F1 (primary tuning objective), AUC-ROC, balanced accuracy, recall, confusion matrices

## Results

| Model | CV F1 | Notes |
| --- | --- | --- |
| Tuned Random Forest | 0.704 | Highest CV F1, but overfit gap 0.088 |
| **Tuned Gradient Boosting** | **0.697** | **Selected for deployment** |

### Why Gradient Boosting over the higher-F1 Random Forest

1. **Generalization.** Overfit gap 0.018 vs 0.088 — roughly 5× lower. Production models that overfit lose accuracy as the data drifts.
2. **Recall.** 84.2% — the highest among the ensemble methods I compared.
3. **Asymmetric cost reasoning.** At any cost ratio above 2:1 (false negative is more expensive than false positive), the recall-maximizing model wins on total expected cost. The 0.007 F1 difference between RF and GB is essentially noise; the generalization and recall advantages are not.

## Threshold Tuning

The deployed Gradient Boosting model was tuned beyond the default 0.5 cutoff:

| Threshold | F1 | Precision | Recall |
| --- | --- | --- | --- |
| 0.50 (default) | 0.6979 | 0.5959 | 0.8422 |
| 0.5767 (optimal F1) | 0.7062 | 0.6511 | 0.7715 |

The threshold itself becomes a strategic lever: revenue-focused hotels can lower the threshold to catch more cancellations (higher recall), while precision-focused scenarios (holiday season, high-value-customer cohorts) can raise it to minimize overbooking risk.

## Business-Strategic Takeaways

- **Lead time is the dominant predictor across models.** Bookings made 200+ days in advance show dramatically higher cancellation rates, anchoring downstream pricing-and-overbooking guidance.
- **Ensemble models outperform linear models**, consistent with non-linear feature interactions in cancellation behavior.
- **The same model serves different operating strategies via threshold choice.** Recall-tilted thresholds for revenue management; precision-tilted thresholds for high-value customer protection.

## A Note on Presentation Design

The final deck opened with an audience-engagement frame: three unlabeled booking profiles, with a question to the audience about which was most likely to cancel.

> **Booking A** — Country: Portugal, Lead time: 200 days, Special requests: 0, Guests: 4 (family)
> **Booking B** — Country: UK, Lead time: 14 days, Special requests: 3, Guests: 2
> **Booking C** — Country: Portugal, Lead time: 90 days, Special requests: 1, Guests: 1

The penultimate slide revealed the model's predictions and reasoning:

> **Booking A — 91.6% cancellation risk.** Drivers: Portuguese booking, long lead time, no special requests. Recommendation: require a deposit or send proactive outreach.
> **Booking B — 1.2% cancellation risk.** Drivers: UK booking, short lead time, high special-request count. Recommendation: no action.
> **Booking C — 9.9% cancellation risk.** Drivers: Portuguese booking, moderate lead time, some special requests. Recommendation: standard reminder.

The framing turned an abstract model into a legible operating tool and made the connection between the analytical work and the operational decision explicit — which is the actual job that "Machine Learning for Business Analytics" is meant to teach.

## Repository Structure

```
.
├── README.md
├── .gitignore
├── BA576 Final Project.ipynb                               # Main analysis notebook
├── BA576 Final Presentation.pdf                            # Final presentation deck
├── Drop deposit_type_ Experimenting & Reasoning (DOC).pdf  # Supporting analysis on dropping deposit_type
└── hotel_bookings.csv                                      # Dataset
```

## Running the Notebook

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook "BA576 Final Project.ipynb"
```

## License

Code is provided for educational and portfolio purposes. The hotel_bookings dataset is publicly available from the source cited above.

---

**Adrian Lin** — BU Questrom B.S. in Business Administration (Business Analytics + Finance), May 2026
[LinkedIn](https://www.linkedin.com/in/adrianlin1) · adrian1@bu.edu
