
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.9400     | 0.9800   |
| ROC AUC          | 0.9972     | 0.9940   |

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
[[47  1]
 [ 1 51]]
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        48
           1       0.98      0.98      0.98        52

    accuracy                           0.98       100
   macro avg       0.98      0.98      0.98       100
weighted avg       0.98      0.98      0.98       100

```

## ROC Curve

![ROC Curve](/intervention_reports/f407_10.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f407_10.0/probability_distributions.png)
