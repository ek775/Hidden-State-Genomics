
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.4800     | 0.9400   |
| ROC AUC          | 0.7664     | 0.9912   |

## Detailed Classification Report (Intervention)

```
[[48  0]
 [52  0]]
              precision    recall  f1-score   support

           0       0.48      1.00      0.65        48
           1       0.00      0.00      0.00        52

    accuracy                           0.48       100
   macro avg       0.24      0.50      0.32       100
weighted avg       0.23      0.48      0.31       100

```

## Detailed Classification Report (Baseline)

```
[[47  1]
 [ 5 47]]
              precision    recall  f1-score   support

           0       0.90      0.98      0.94        48
           1       0.98      0.90      0.94        52

    accuracy                           0.94       100
   macro avg       0.94      0.94      0.94       100
weighted avg       0.94      0.94      0.94       100

```

## ROC Curve

![ROC Curve](/intervention_reports/f407_1.2/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f407_1.2/probability_distributions.png)
