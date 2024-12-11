import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


def cluster_3d(X, y, labels, features, colors):
    df = X.copy()
    df['stroke'] = y

    df['cluster'] = labels

    x_feature = features[0]
    y_feature = features[1]
    z_feature = features[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), subplot_kw={'projection': '3d'})

    ax1.scatter(df[x_feature], df[y_feature], df[z_feature], c=df['cluster'].map(colors), s=50)
    ax1.set_xlabel(x_feature)
    ax1.set_ylabel(y_feature)
    ax1.set_zlabel(z_feature)
    ax1.set_title('KMeans Clustering (3D Visualization)')

    ax2.scatter(df[x_feature], df[y_feature], df[z_feature], c=df['cluster'].map(colors), s=50)
    ax2.set_xlabel(x_feature)
    ax2.set_ylabel(y_feature)
    ax2.set_zlabel(z_feature)
    ax2.set_title('KMeans Clustering (3D Visualization)')
    ax2.view_init(elev=90, azim=0)

    plt.show()
    
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