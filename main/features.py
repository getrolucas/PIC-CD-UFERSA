import pandas as pd
import numpy as np
import holidays
from statsmodels.tsa.seasonal import MSTL #type: ignore


__all__ = ['add_features']


def _feriados(dia: str | pd.Timestamp) -> int:
    """Valida se uma data é feriado nacional.

    Args:
        dia (str): Dia para validação.

    Returns:
        pd.DataFrame: 1 (sim) ou 0 (não) para cada localidade.
    """
    dia = pd.to_datetime(dia)
    feriados_br = holidays.country_holidays('BR')

    return 1 if dia in feriados_br else 0


def _outliers(df : pd.DataFrame, target_col : str = 'y') -> pd.DataFrame:
    """Calcula outliers com base em 3*IQR dos dados 
    e atribui ao df duas colunas, uma para outliers de mínimo e outra para máximo.

    Args:
        df (pd.DataFrame): Dados originais para calcular e inserir colunas.
        target_col (str, optional): Coluna com dados usados para cálculo. Padrão é 'y'.

    Returns:
        pd.DataFrame: DataFrame expendido com colunas indicando outliers.
    """
    stl = MSTL(df[target_col], periods=[7, 365])
    res = stl.fit()
    q1 = res.resid.quantile(0.25)
    q3 = res.resid.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 3 * iqr
    limite_sup = q3 + 3 * iqr

    df['outlier_min'] = np.where(df[target_col] < limite_inf, 1, 0)
    df['outlier_max'] = np.where(df[target_col] > limite_sup, 1, 0)

    return df


def add_features(
        df : pd.DataFrame,
        df_type : str,
        time_col : str = 'ds',
) -> pd.DataFrame:
    """Adiciona features de calendário e outliers ao dataframe.

    Args:
        df (pd.DataFrame): DataFrame contendo coluna de data no tipo datetime 
            e coluna de valores para calculo de outliers.
        df_type (str): Se o DataFrame é de treino ou teste.
            Implica em atribuir outliers se for treino, ou não atribuir se for teste.
        time_col (str, optional): Nome da coluna de datas. Padrão é 'ds'.
        target_col (str, optional): Nome da coluna target. Padrão é 'y'.

    Returns:
        pd.DataFrame: DataFrame expandido com as features.
    """
    df['feriado'] = df[time_col].apply(_feriados)
    df['day_of_week'] = df[time_col].dt.day_of_week
    df['week'] = df[time_col].dt.isocalendar().week.astype(str)
    df['month'] = df[time_col].dt.month
    df['quarter'] = df[time_col].dt.quarter
    
    actions = {
        'train': lambda df: _outliers(df),
        'test': lambda df: df.assign(outlier_min=0, outlier_max=0)
    }
    
    return actions.get(df_type, lambda df: df)(df)