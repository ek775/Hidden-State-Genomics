
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.5030     | 0.9900   |
| ROC AUC          | 0.9847     | 0.9926   |

## Detailed Classification Report (Intervention)

```
              precision    recall  f1-score   support

           0       0.50      1.00      0.67       498
           1       0.86      0.01      0.02       502

    accuracy                           0.50      1000
   macro avg       0.68      0.50      0.35      1000
weighted avg       0.68      0.50      0.34      1000

```
![Confusion Matrix (Intervention)](/intervention_reports/f1422_m0.1_a10.0/confusion_matrix_intervention.png)

## Detailed Classification Report (Baseline)

```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99       498
           1       0.98      1.00      0.99       502

    accuracy                           0.99      1000
   macro avg       0.99      0.99      0.99      1000
weighted avg       0.99      0.99      0.99      1000

```
![Confusion Matrix (Baseline)](/intervention_reports/f1422_m0.1_a10.0/confusion_matrix_baseline.png)

## ROC Curve

![ROC Curve](/intervention_reports/f1422_m0.1_a10.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/intervention_reports/f1422_m0.1_a10.0/probability_distributions.png)
