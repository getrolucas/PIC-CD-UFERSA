import pandas as pd
from matplotlib import pyplot as plt
from typing import Union
import math


__all__ = ['plot_time_series', 'plot_hist']


def plot_time_series(
    df: pd.DataFrame,
    data_cols : list,
    id_col: str = 'unique_id', 
    ids: list | None = None,
    grid: tuple | None = None, 
    figsize: tuple = (12, 8),
) -> None:
    """Plot das séries temporais e das previsões.

    Args:
        df (pd.DataFrame): Dados com valores históricos e previsões.
        data_cols (str): Nomes das colunas a serem adicionadas ao mesmo plot.
        id_col (str): Coluna de identificação de cada série. Padrão é 'unique_id'.
        ids (Union[list, None], optional): Ids a serem plotados. Implica na quantidade de plots. Padrão é None.
        grid (tuple, optional): Matriz de plots tipo nxm. Padrão é n/2.
        figsize (tuple, optional): Tamanho do plot. Padrão é (12, 8).
    """
    if ids is None:
        ids = df[id_col].unique().tolist()

    n_plots = len(ids)

    if grid is None:
        rows = math.ceil(n_plots / 2)
        cols = 2
        grid = (rows, cols)
    
    plt.figure(figsize=figsize)
    
    plot_n = 1
    
    for id in ids:
        df_id = df[df[id_col] == id]
    
        plt.subplot(grid[0], grid[1], plot_n)
        for col in data_cols:
            plt.plot(df_id[col])
        plt.title(f'unique_id={id}', fontdict={'size': 10})
        plot_n += 1

    plt.tight_layout()
    plt.show()


def plot_hist(
    df: pd.DataFrame, 
    data_col: str, 
    id_col: str = 'unique_id', 
    ids: Union[list, None] = None,
    grid: tuple | None = None, 
    figsize: tuple = (12, 8)
) -> None:
    """Plot da distribuição dos dados.

    Args:
        df (pd.DataFrame): Dados com coluna de ids e valores para distribuição.
        data_col (list): Coluna a ser plotadas.
        id_col (str): Coluna de identificação de cada série. Padrão é 'unique_id'.
        ids (Union[list, None], optional): Ids a serem plotados. Implica na quantidade de plots. Padrão é None.
        grid (tuple, optional): Matriz de plots tipo nxm. Padrão é n/2.
        figsize (tuple, optional): Tamanho do plot. Padrão é (12, 8).
    """
    if ids is None:
        ids = df[id_col].unique().tolist()

    n_plots = len(ids)

    if grid is None:
        rows = math.ceil(n_plots / 2)
        cols = 2
        grid = (rows, cols)
    
    plt.figure(figsize=figsize)
    
    plot_n = 1
    
    for id in ids:
        df_id = df[df[id_col] == id]
        
        for col in data_col:
            plt.subplot(grid[0], grid[1], plot_n)
            plt.hist(df_id[col], bins=20, edgecolor='black', alpha=0.7)
            plt.title(f'unique_id={id}', fontdict={'size': 10})
            plot_n += 1
    
    plt.tight_layout()
    plt.show()