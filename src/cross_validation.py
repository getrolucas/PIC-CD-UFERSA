from typing import Generator, Tuple
import pandas as pd

def time_series_cv(
        df: pd.DataFrame, 
        n_folds: int, 
        window: int
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Gera DataFrames de treino e validação para validação cruzada em séries temporais.

    Args:
        df (pd.DataFrame): DataFrame contendo a série temporal.
        n_folds (int): Número de divisões (folds) para validação cruzada.
        window (int): Tamanho da janela para cada fold.

    Yields:
        Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]: 
        DataFrames de treino e validação para cada iteração em n_folds.
    """
    total_size = len(df)
    initial_train_end = total_size - window * n_folds

    for fold in range(n_folds):
        train_end = initial_train_end + fold * window
        val_start = train_end
        val_end = val_start + window
        
        yield df.iloc[:train_end], df.iloc[val_start:val_end]
