import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import os

def plot_roc_auc(model, X_train, y_train, X_test, y_test, model_name="Model"):
    
    """
    Genera la ROC Curve per a qualsevol model"

    Parametres:
    - model: Classificador
    - X_train: Set d'entrenament.
    - y_train: Etiquetes d'entrenament.
    - X_test: Set de test.
    - y_test: Etiquetes de test.
    - model_name: Etiqueta del model que es mostrarà al gràfic

    Sortida:
    - ROC Curve i AUC score
    """
    #Carpeta per guardar els resultats
    output_dir = "ROC_Data"
    os.makedirs(output_dir, exist_ok=True) 
    
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]  

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    #Guardar els arrays
    np.save(os.path.join(output_dir, f"{model_name}_fpr.npy"), fpr)
    np.save(os.path.join(output_dir, f"{model_name}_tpr.npy"), tpr)
    np.save(os.path.join(output_dir, f"{model_name}_thresholds.npy"), thresholds)
    np.save(os.path.join(output_dir, f"{model_name}_auc.npy"), np.array([auc_score]))
    
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
    
    
 