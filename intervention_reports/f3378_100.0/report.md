
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.9500     | 0.9800   |
| ROC AUC          | 0.9724     | 0.9884   |

## Detailed Classification Report (Intervention)

```
[[45  3]
 [ 2 50]]
              precision    recall  f1-score   support

           0       0.96      0.94      0.95        48
           1       0.94      0.96      0.95        52

    accuracy                           0.95       100
   macro avg       0.95      0.95      0.95       100
weighted avg       0.95      0.95      0.95       100

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

![ROC Curve](/intervention_reports/f3378_100.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f3378_100.0/probability_distributions.png)
