from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def metrics(y_test: list, y_pred: list, print_metrics: bool = False) -> Tuple[float, float, float, float, list]:
    """
    Donat els valors reals i predits, retorna les mètriques d'avaluació.

    :param y_test: Valors reals.
    :param y_pred: Valors predits.
    :param print_metrics: Si és True, imprimeix les mètriques.
    :return: Retorna un tuple amb cinc elements. El primer element és l'accuracy, el segon és la precisió, el tercer és el recall, el quart és el F1 score, i el cinquè és la matriu de confusió.
    """

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    if print_metrics:
        print(f"Accuracy: {acc:.4f}",)
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0','1'], yticklabels=['0','1'])   
        plt.ylabel('True')
        plt.title('Confusion Matrix Heatmap')

        plt.tight_layout()
        plt.show()

    return acc, prec, rec, f1, cm