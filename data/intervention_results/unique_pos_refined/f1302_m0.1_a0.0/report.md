
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.5180     | 0.8830   |
| ROC AUC          | 0.9339     | 0.9891   |

## Detailed Classification Report (Intervention)

```
              precision    recall  f1-score   support

           0       0.51      1.00      0.67       498
           1       1.00      0.04      0.08       502

    accuracy                           0.52      1000
   macro avg       0.75      0.52      0.38      1000
weighted avg       0.76      0.52      0.37      1000

```
![Confusion Matrix (Intervention)](/intervention_reports/f1302_m0.1_a0.0/confusion_matrix_intervention.png)

## Detailed Classification Report (Baseline)

```
              precision    recall  f1-score   support

           0       0.82      0.98      0.89       498
           1       0.98      0.78      0.87       502

    accuracy                           0.88      1000
   macro avg       0.90      0.88      0.88      1000
weighted avg       0.90      0.88      0.88      1000

```
![Confusion Matrix (Baseline)](/intervention_reports/f1302_m0.1_a0.0/confusion_matrix_baseline.png)

## ROC Curve

![ROC Curve](/intervention_reports/f1302_m0.1_a0.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f1302_m0.1_a0.0/probability_distributions.png)
