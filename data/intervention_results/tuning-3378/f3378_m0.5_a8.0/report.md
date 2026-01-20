
# Intervention Report

| Metric           | Intervention | Baseline |
|------------------|--------------|----------|
| Accuracy         | 0.9840     | 0.9110   |
| ROC AUC          | 0.9969     | 0.9978   |

## Detailed Classification Report (Intervention)

```
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       498
           1       0.99      0.98      0.98       502

    accuracy                           0.98      1000
   macro avg       0.98      0.98      0.98      1000
weighted avg       0.98      0.98      0.98      1000

```
![Confusion Matrix (Intervention)](/data/tuning-3378/f3378_m0.5_a8.0/confusion_matrix_intervention.png)

## Detailed Classification Report (Baseline)

```
              precision    recall  f1-score   support

           0       0.85      1.00      0.92       498
           1       1.00      0.83      0.90       502

    accuracy                           0.91      1000
   macro avg       0.92      0.91      0.91      1000
weighted avg       0.92      0.91      0.91      1000

```
![Confusion Matrix (Baseline)](/data/tuning-3378/f3378_m0.5_a8.0/confusion_matrix_baseline.png)

## ROC Curve

![ROC Curve](/data/tuning-3378/f3378_m0.5_a8.0/roc_curve.png)

## Predicted Probability Distributions

![Probability Distributions](/data/tuning-3378/f3378_m0.5_a8.0/probability_distributions.png)
