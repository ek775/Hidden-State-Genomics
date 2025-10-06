
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.9400     | 0.9900   |
| ROC AUC          | 1.0000     | 0.9992   |

## Detailed Classification Report (Intervention)

```
[[42  6]
 [ 0 52]]
              precision    recall  f1-score   support

           0       1.00      0.88      0.93        48
           1       0.90      1.00      0.95        52

    accuracy                           0.94       100
   macro avg       0.95      0.94      0.94       100
weighted avg       0.95      0.94      0.94       100

```

## Detailed Classification Report (Baseline)

```
[[48  0]
 [ 1 51]]
              precision    recall  f1-score   support

           0       0.98      1.00      0.99        48
           1       1.00      0.98      0.99        52

    accuracy                           0.99       100
   macro avg       0.99      0.99      0.99       100
weighted avg       0.99      0.99      0.99       100

```

## ROC Curve

![ROC Curve](/intervention_reports/f7030_10.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f7030_10.0/probability_distributions.png)
