
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.6100     | 0.9800   |
| ROC AUC          | 0.9712     | 0.9804   |

## Detailed Classification Report (Intervention)

```
[[ 9 39]
 [ 0 52]]
              precision    recall  f1-score   support

           0       1.00      0.19      0.32        48
           1       0.57      1.00      0.73        52

    accuracy                           0.61       100
   macro avg       0.79      0.59      0.52       100
weighted avg       0.78      0.61      0.53       100

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

![ROC Curve](/intervention_reports/f7030_100.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f7030_100.0/probability_distributions.png)
