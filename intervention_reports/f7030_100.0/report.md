
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.4800     | 0.9900   |
| ROC AUC          | 0.7953     | 1.0000   |

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
 [ 0 52]]
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        48
           1       0.98      1.00      0.99        52

    accuracy                           0.99       100
   macro avg       0.99      0.99      0.99       100
weighted avg       0.99      0.99      0.99       100

```

## ROC Curve

![ROC Curve](/intervention_reports/f7030_100.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f7030_100.0/probability_distributions.png)
