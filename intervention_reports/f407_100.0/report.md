
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.9500     | 1.0000   |
| ROC AUC          | 0.9838     | 1.0000   |

## Detailed Classification Report (Intervention)

```
[[46  2]
 [ 3 49]]
              precision    recall  f1-score   support

           0       0.94      0.96      0.95        48
           1       0.96      0.94      0.95        52

    accuracy                           0.95       100
   macro avg       0.95      0.95      0.95       100
weighted avg       0.95      0.95      0.95       100

```

## Detailed Classification Report (Baseline)

```
[[48  0]
 [ 0 52]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        48
           1       1.00      1.00      1.00        52

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100

```

## ROC Curve

![ROC Curve](/intervention_reports/f407_100.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f407_100.0/probability_distributions.png)
