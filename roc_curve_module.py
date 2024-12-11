import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_auc(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Generate and plot the ROC Curve and AUC for any classifier model.

    Parameters:
    - model: The classifier object (must implement fit and predict_proba methods).
    - X_train: Training feature set.
    - y_train: Training labels.
    - X_test: Testing feature set.
    - y_test: Testing labels.
    - model_name: String to label the model in the plot.

    Output:
    - Displays the ROC Curve and prints the AUC score.
    """
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class

    # Compute ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    # Plot the ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', label=f'{model_name} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    plt.title(f'ROC Curve: {model_name}')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    print(f"{model_name} AUC Score: {auc_score:.2f}")