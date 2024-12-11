import pandas as pd
from typing import Tuple

def limit_rows(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Donat un DataFrame de Pandas, retorna un nou DataFrame amb un nombre limitat de files.

    :param df: DataFrame de Pandas.
    :param limit: Número de files a retornar.
    :return: Retorna un nou DataFrame de Pandas amb un nombre limitat de files.
    """
    return df.head(limit)


def balance_data(df: pd.DataFrame, target_column: str, num_limit: int) -> pd.DataFrame:
    """
    Donat un DataFrame de Pandas, retorna un nou DataFrame amb les classes balancejades, limitant el nombre de files per classe.

    :param df: DataFrame de Pandas.
    :param target_column: Nom de la columna objectiu.
    :return: Retorna un nou DataFrame de Pandas equilibrat.
    """
    df_0 = df[df[target_column] == 0].sample(n=num_limit, random_state=42)
    df_1 = df[df[target_column] == 1].sample(n=num_limit, random_state=42)
    return pd.concat([df_0, df_1]).reset_index(drop=True)


def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Donat un path d'un fitxer CSV, retorna les dades en format DataFrame de Pandas.

    :param file_path: Path del fitxer CSV.
    :return: Retorna un tuple amb dos elements. El primer element és un DataFrame de Pandas amb les dades d'entrada, i el segon element és un DataFrame de Pandas amb les dades de sortida.
    """
    df = pd.read_csv(file_path)  # Carreguem les dades del fitxer CSV

    # Per columnes amb valors binaris, fem un map per convertir-los a 0 i 1

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})
    df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})

    df["obesity"] = df["bmi"].map(lambda x: 1 if x >= 30 else 0)

    # Per columnes categoriques, fem un one-hot encoding
    df = pd.get_dummies(df, columns=["smoking_status", "work_type"])

    df = df.dropna()  # Eliminem files amb valors nuls


    #positius = df["stroke"].sum()
    #df = balance_data(df, "stroke", positius)  # Equilibrem les dades


    # Separem entre dades d'entrada i sortida
    X = df.drop(columns=["id", "stroke"])
    y = df["stroke"]

    smote = SMOTE(random_state = 10)
    X, y = smote.fit_resample(X, y)

    return X, y


if __name__ == "__main__":
    X, y = load_data("data/healthcare-dataset-stroke-data.csv")
    print(X.head())
    print(y.head())
    print("Mean age for stroke=1:", X[y == 1]["age"].mean())
    print("Mean age for stroke=0:", X[y == 0]["age"].mean())
