
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.4800     | 0.9800   |
| ROC AUC          | 0.7620     | 0.9804   |

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
[[46  2]
 [ 0 52]]
              precision    recall  f1-score   support

           0       1.00      0.96      0.98        48
           1       0.96      1.00      0.98        52

    accuracy                           0.98       100
   macro avg       0.98      0.98      0.98       100
weighted avg       0.98      0.98      0.98       100

```

## ROC Curve

![ROC Curve](/intervention_reports/f8161_100.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f8161_100.0/probability_distributions.png)
