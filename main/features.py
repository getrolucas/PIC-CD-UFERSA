import pandas as pd
import numpy as np
import holidays
from statsmodels.tsa.seasonal import MSTL
from sklearn.linear_model import LassoLarsCV

__all__ = ['add_features', 'add_lagged_features', 'feature_selection']


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
    df['day'] = df[time_col].dt.day
    df['month_end'] = df[time_col].dt.is_month_end.astype(int)
    df['feriado'] = df[time_col].apply(_feriados)
    df['day_of_week'] = df[time_col].dt.day_of_week
    df['week'] = df[time_col].dt.isocalendar().week.astype(int)
    df['month'] = df[time_col].dt.month
    df['quarter'] = df[time_col].dt.quarter

    actions = {
        'train': lambda df: _outliers(df),
        'test': lambda df: df.assign(outlier_min=0, outlier_max=0)
    }
    
    return actions.get(df_type, lambda df: df)(df)


def add_lagged_features(
    df : pd.DataFrame, 
    features : list, 
    lags : list
):
    """
    Adiciona novas features que são lags das features passadas. 

    Args:
        df (pd.DataFrame): DataFrame contendo as features.
        features (list): Features para adicionar lags.
        lags (Union[list, range]): Lags a serem adicionados. Ex.: [-1, 1]
    Returns:
        pd.DataFrame: DataFrame com as features adicionadas. 
    """
    df = df.copy()
    for feature in features:
        for lag in lags:
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag).fillna(0)
    return df


def feature_selection(
        X : np.ndarray | pd.DataFrame, 
        y : np.ndarray | pd.Series
) -> list:
    """Feature selection usando penalização via regressão de Lasso.

    Args:
        X (np.narray | pd.DataFrame): Variáveis explicativas/regressoras.
        y (np.narray | pd.DataFrame): Variável resposta.

    Returns:
        list: Features selecionadas.
    """
    lasso = LassoLarsCV(cv=5, max_n_alphas=10)
    res = lasso.fit(X=X, y=y)
    features = res.feature_names_in_[lasso.coef_!=0]  
    
    return list(features)
