import pandas as pd
from src.features import decompose
from outliers import smirnov_grubbs as esd


def _detector(
        df : pd.DataFrame, 
        time_col : str = 'ds',
        target_col : str = 'y',
        id_col : str = 'unique_id',
        period : int = 7
) -> pd.DataFrame:
    """Função para detectar outliers em cada série individualmente.
    Será usada no wrapper que aplicará para todos as séries.
    """
    _df = df.reset_index(drop=True)
    max_index = esd.max_test_indices(
        decompose(_df, time_col, target_col, id_col, period).resid.to_numpy()
    )
    min_index = esd.min_test_indices(
        decompose(_df, time_col, target_col, id_col, period).resid.to_numpy()
    )
    
    _df.loc[max_index, 'max_outliers'] = 1
    _df.loc[min_index, 'min_outliers'] = 1

    return _df.fillna(0)

def detect_outliers(
    df : pd.DataFrame, 
    time_col : str = 'ds', 
    target_col : str = 'y',
    id_col : str = 'unique_id',
    period : int = 7
) -> pd.DataFrame:  
    """Detecção de outliers via método de Grubbs aplicado 
    recursivamente para valores extremos mínimos e máximos em
    resíduos de séries temporais decompostas.
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm

    Args:
        df (pd.DataFrame): Dados com séries temporais para detecção dos outliers.
        time_col (str, optional): Coluna com valores de tempo. Padrão é 'ds'.
        target_col (str, optional):  Nome da coluna com a série a ser decomposta. Padrão é 'y'.
        id_col (str, optional): Nome da coluna que identifica séries únicas. Padrão é 'unique_id'.

    Returns:
        pd.DataFrame: Dados originais adicionados das colunas `max_outliers` e `min_outliers`.
    """
    dfs = []

    for i in df[id_col].unique():
        dfs.append(
            _detector(df.loc[df[id_col] == i], time_col, target_col, id_col, period)
        )
    
    return pd.concat(dfs, ignore_index=True)