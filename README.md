[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17032340&assignment_repo_type=AssignmentRepo)


# Predicció d'Atacs de Cor utilitzant Machine Learning

## Descripció General
Aquest projecte analitza dades mèdiques per resoldre diversos problemes com:
- **Predicció d'atacs de cor.**
- **Classificació de risc.**
- **Agrupació per similitud de pacients.**

## Datasets
Aquest dataset conté informació com:
- **id:** Identificador únic del pacient.
- **gender:** Sexe del pacient.
- **age:** Edat del pacient.
- **hypertension:** Historial d'hipertensió.
- **heart_disease:** Historial de malalties del cor.
- **ever_married:** Estat civil del pacient.
- **work_type:** Tipus de treball del pacient.
- **Residence_type:** Tipus de residència del pacient.
- **avg_glucose_level:** Nivell mitjà de glucosa en sang.
- **bmi:** Índex de massa corporal.
- **smoking_status:** Estat de tabaquisme del pacient.
- **stroke:** Historial d'ictus.

## Fases del Projecte i Models
### 1. Regressió Logística: Predicció d'Atacs de Cor
- **Problema:** Predir la probabilitat de patir un atac de cor basant-se en factors de risc individuals.
- **Model:** Regressió Logística.
- **Entrades i Sortides:** Factors de risc com a entrada, probabilitat d'atac de cor com a sortida.
- **Mètriques:** AUC-ROC, Accuracy, Precision, Recall.

### 2. Clustering: Agrupació per Similitud de Pacients
- **Problema:** Agrupar pacients similars segons factors de risc.
- **Model:** K-Means.
- **Visualització:** Representació en 3D o 2D utilitzant reducció de dimensionalitat.
- **Resultat esperat:** Identificar grups com "Pacients amb alt risc" o "Pacients amb baix risc".

### 3. Ampliacions Opcionals
- **Predicció de Tractaments:** Estimar l'eficàcia de diferents tractaments basant-se en les característiques del pacient. (S'hauria de fer servir un altre dataset)

## Recursos
- [Dataset de Malalties del Cor](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset )


### Feedback:

- SMOTE mirar parametres (treure) 
- Data balancing nomes train (test opcional)
- Class weight logistic regression 
- Fer la ROC CURVE a tots els models 
- Fer la comparació de les ROC Curves de cada model
- Prova amb menys caracteristiques
- Opcional: Fer servir LaTex per informe