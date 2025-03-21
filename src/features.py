import pandas as pd
import numpy as np
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LassoCV

__all__ = [
    'decompose', 
    'add_features', 
    'add_lagged_features', 
    'feature_selection'
]

def _decomposer(
        df : pd.DataFrame, 
        time_col : str = 'ds', 
        target_col : str = 'y',
        period : int = 7
    ) -> pd.DataFrame:
    _df = df.copy()

    res = seasonal_decompose(
        x=_df.set_index(time_col)[target_col], 
        model='additive', 
        two_sided=True, 
        period=period,
        extrapolate_trend='freq'
    )

    _df = _df.assign(
        trend = res.trend.values,
        seasonal = res.seasonal.values,
        resid = res.resid.values,
    )
    return _df


def decompose(
    df : pd.DataFrame, 
    time_col : str = 'ds', 
    target_col : str = 'y',
    id_col : str = 'unique_id',
    period : int = 7
) -> pd.DataFrame:  
    """Decompõe múltiplas séries temporais usando `statsmodels.tsa.seasonal.seasonal_decompose`.

    Args:
        df (pd.DataFrame): Dados de uma ou mais séries temporais.
        time_col (str, optional): Coluna com valores de tempo. Padrão é 'ds'.
        y_col (str, optional): Nome da coluna com a série a ser decomposta. Padrão é 'y'.
        id_col (str, optional): Nome da coluna que identifica séries únicas. Padrão é 'unique_id'.

    Returns:
        pd.DataFrame: Dados originais com novas colunas dos termos de tendência, sazonalidade e resíduo adicionadas.
    """
    decomposed_df = []

    for i in df[id_col].unique():
        decomposed_df.append(
            _decomposer(df.loc[df[id_col] == i], time_col=time_col, target_col=target_col, period=period)
        )
    
    return pd.concat(decomposed_df, ignore_index=True)


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


def add_calendar_features(
        df : pd.DataFrame,
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

    return df


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


def lasso_feature_selection(
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
    lasso = LassoCV()
    res = lasso.fit(X=X, y=y)
    features = res.feature_names_in_[lasso.coef_!=0]  
    
    return list(features)

def add_trend(
        train_df : pd.DataFrame,
        test_df : pd.DataFrame
) -> pd.DataFrame:
    """Adiciona índice t para modelar tendência linear.

    Args:
        train_df (pd.DataFrame): Dados de treino.
        test_df (pd.DataFrame): Dados de teste.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): Dados de treino e teste com coluna `trend` adicionada.
    """
    train_df['trend'] = train_df.index
    test_df['trend'] = test_df.index + train_df.index.max() + 1
    return train_df, test_df