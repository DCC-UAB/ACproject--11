import pandas as pd
from typing import Tuple


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Donat un path d'un fitxer CSV, retorna les dades en format DataFrame de Pandas.

    :param file_path: Path del fitxer CSV.
    :return: Retorna un tuple amb dos elements. El primer element és un DataFrame de Pandas amb les dades d'entrada, i el segon element és un DataFrame de Pandas amb les dades de sortida.
    """

    df = pd.read_csv(file_path) # Carreguem les dades del fitxer CSV

    # Per columnes amb valors binaris, fem un map per convertir-los a 0 i 1
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})
    df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})

    # Per columnes categoriques, fem un one-hot encoding
    df = pd.get_dummies(df, columns=["smoking_status", "work_type"])
    
    df = df.dropna() # Eliminem files amb valors nuls

    # Separem entre dades d'entrada i sortida
    X = df.drop(columns=["id", "stroke"])
    y = df["stroke"]

    return X, y


if __name__ == "__main__":
    X, y = load_data("data/healthcare-dataset-stroke-data.csv")
    print(X.head())
    print(y.head())
