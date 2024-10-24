import numpy as np
import pandas as pd


def mae(y, y_pred) -> np.float64:
    """
    Erro absoluto médio (Mean Absolute Error [MAE])
    """
    return np.mean(np.abs(y - y_pred))


def mse(y, y_pred) -> np.float64:
    """
    Erro quadrático médio (Mean Squared Error [MSE])
    """
    return np.mean((y - y_pred) ** 2)


def rmse(y, y_pred) -> np.float64:
    """
    Raiz do Erro Quadrático Médio (Root Mean Squared Error [RMSE])
    """
    return np.sqrt(mse(y, y_pred))


def r2(y, y_pred) -> np.float64:
    """
    Coeficiente de determinação (Coefficient of Determination [R²])
    """
    dnm = np.sum((y - np.mean(y)) ** 2)
    nm = np.sum((y - y_pred) ** 2)

    return 1 - np.divide(nm, dnm)


class Evaluation:
    """
    Classe para avaliação de modelos de regressão, fornecendo as principais métricas de desempenho:
    - MAE: Mean Absolute Error (Erro Absoluto Médio)
    - MSE: Mean Squared Error (Erro Quadrático Médio)
    - RMSE: Root Mean Squared Error (Raiz do Erro Quadrático Médio)
    - R²: Coeficiente de Determinação (R-squared)
    
    Construção das métricas baseada no texto: *Métricas para Regressão: Entendendo as métricas R², MAE, MAPE, MSE e RMSE.*
    Disponível em: https://medium.com/data-hackers/prevendo-n%C3%BAmeros-entendendo-m%C3%A9tricas-de-regress%C3%A3o-35545e011e70
    
    Parameters
    ---
    y : pd.Series | np.array
        Valores reais observados.
    y_pred: pd.Series | np.array
        Valores preditos pelo modelo.
    id_col : str | None
        Coluna de identificação de cada série. Padrão é 'unique_id'.

    
    Methods
    ---
    summary():
        Retorna um DataFrame com todas as métricas de avaliação.
    """

    # TODO: Comportar vários modelos. Pensar numa forma intuitiva de mostrar.
    
    def __init__(
            self, 
            df : pd.DataFrame,
            y_col : str = 'y', 
            y_pred_col : str = 'y_pred',
            unique_id : str | None = 'unique_id'
    ) -> None:
        
        self.df = df
        self.y_col = y_col
        self.y_pred_col = y_pred_col
        self.unique_id = unique_id
        

    def summary(self) -> pd.DataFrame:
        """
        Returns
        ---
        pd.DataFrame com sumário de desempenho do modelo.
        """
        metrics = [mae, mse, rmse, r2]
        if self.unique_id is None:
            res = {
                f.__name__ : f(self.df[self.y_col], self.df[self.y_pred_col]) for f in metrics
            }
            return pd.DataFrame(res, index=[0])
        else:
            all_series_res = {}
            for id in self.df[self.unique_id].unique():
                temp_df = self.df.query(f"{self.unique_id} == @id")
                all_series_res[id] = {
                    f.__name__ : f(temp_df[self.y_col], temp_df[self.y_pred_col])for f in metrics
                }
            return pd.DataFrame(all_series_res).T
            