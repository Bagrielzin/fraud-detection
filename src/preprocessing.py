import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega o dataset e faz validações básicas."""
    df = pd.read_csv(filepath)
    required_cols = {"Time", "Amount", "Class"}
    assert required_cols.issubset(df.columns), f"Colunas ausentes: {required_cols - set(df.columns)}"
    print(f"Dataset carregado: {df.shape[0]:,} linhas, {df.shape[1]} colunas")
    print(f"Fraudes: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas features a partir de Time e Amount."""
    df = df.copy()

    # Hora do dia (o dataset cobre ~48h)
    df["hour_of_day"] = (df["Time"] / 3600) % 24
    df["hour_of_day"] = df["hour_of_day"].round(1)

    # Transformação logarítmica do valor
    df["log_amount"] = np.log1p(df["Amount"])

    # Z-score do Amount
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

    # Flag de transação noturna
    df["is_night"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)

    return df


def split_data(
    df: pd.DataFrame,
    target: str = "Class",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """Divide os dados em treino, validação e teste de forma estratificada."""
    X = df.drop(columns=[target])
    y = df[target]

    # Separar teste primeiro
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Separar validação do restante
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    print(f"Divisão dos dados:")
    print(f"Treino:    {X_train.shape[0]:>7,} amostras | fraudes: {y_train.sum():,}")
    print(f"Validação: {X_val.shape[0]:>7,} amostras | fraudes: {y_val.sum():,}")
    print(f"Teste:     {X_test.shape[0]:>7,} amostras | fraudes: {y_test.sum():,}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_preprocessor(cols_to_scale: list) -> RobustScaler:
    """Retorna um RobustScaler configurado (resistente a outliers)."""
    return RobustScaler()


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """Aplica SMOTE apenas no conjunto de treino para balancear as classes."""
    smote = SMOTE(random_state=random_state, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE aplicado:")
    print(f"Antes:  {y_train.sum():,} fraudes / {len(y_train):,} total")
    print(f"Depois: {y_res.sum():,} fraudes / {len(y_res):,} total")
    return X_res, y_res
