from sklearn.metrics import confusion_matrix

y_true = [0, 0, 1, 1]
y_pred = [0, 0, 0, 1]

cm = confusion_matrix(y_true, y_pred)
print(cm)