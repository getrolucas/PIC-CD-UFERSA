import pandas as pd
from typing import Tuple


__all__ = ['ts_train_test_split']


def _splitter(
        df : pd.DataFrame, 
        train_size: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[:round(len(df) * train_size), :]
    test_df = df.drop(train_df.index)
    return train_df, test_df


def ts_train_test_split(
        df : pd.DataFrame,
        train_size : float,
        id_col : str = 'unique_id'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separação entre dados de treino e teste para série temporal.

    Args:
        df (pd.DataFrame): Dados de uma ou mais séries temporais.
        train_size (float): Deve estar entre 0.0 e 1.0 e representa a proporção do dados de treino.
        id_col (str, optional): Coluna de identificação de cada série. Padrão é 'unique_id'.

    Returns:
        train_df (pd.DataFrame): DataFrame para treinar o modelo.
        
        test_df (pd.DataFrame):  DataFrame para avaliar a capacidade de previsão do modelo.
    """
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    for i in df[id_col].unique():
        res = _splitter(df.loc[df[id_col] == i], train_size)
        train_df = pd.concat((train_df, res[0]), ignore_index=True)
        test_df = pd.concat((test_df, res[1]), ignore_index=True)
    return train_df, test_df
