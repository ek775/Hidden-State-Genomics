
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.9700     | 0.9600   |
| ROC AUC          | 0.9772     | 0.9760   |

## Detailed Classification Report (Intervention)

```
[[45  3]
 [ 0 52]]
              precision    recall  f1-score   support

           0       1.00      0.94      0.97        48
           1       0.95      1.00      0.97        52

    accuracy                           0.97       100
   macro avg       0.97      0.97      0.97       100
weighted avg       0.97      0.97      0.97       100

```

## Detailed Classification Report (Baseline)

```
[[45  3]
 [ 1 51]]
              precision    recall  f1-score   support

           0       0.98      0.94      0.96        48
           1       0.94      0.98      0.96        52

    accuracy                           0.96       100
   macro avg       0.96      0.96      0.96       100
weighted avg       0.96      0.96      0.96       100

```

## ROC Curve

![ROC Curve](/intervention_reports/f7030_3.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f7030_3.0/probability_distributions.png)
