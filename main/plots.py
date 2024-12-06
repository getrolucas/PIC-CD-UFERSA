import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Union
import math


__all__ = ['plot_time_series', 'plot_hist', 'plot_metrics']


def plot_time_series(
    df: pd.DataFrame,
    data_cols : list,
    time_col : str = 'ds',
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
            plt.plot(df_id[time_col], df_id[col])
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


def _minmax(row: pd.Series) -> pd.Series:
    min_val, max_val = row.min(), row.max()
    return (row - min_val) / (max_val - min_val)


def plot_metrics(
        df: pd.DataFrame, 
        hl_col: str | None = None, 
        figsize: tuple = (6, 6),
        annot_kws : dict = {"fontsize": 10}
    ) -> None:
    """
    Plota as métricas de desempenho dos modelos por série em um heatmap 
    possibilitando comparar visualmente qual modelo foi melhor.

    Args:
        df (pd.DataFrame): DataFrame com os dados a serem plotados.
        hl_col (str): Nome da coluna a ser destacada.
        figsize (tuple): Tamanho da figura.
        annot_kws (dict): Kwargs para ajuste da fonte da anotação.
    """
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        data=df.T.apply(_minmax).T,
        annot=df,
        fmt=".2f",
        cbar=False,
        cmap='Blues',
        annot_kws=annot_kws
    )

    if hl_col:
        if hl_col not in df.columns:
            raise ValueError(f"A coluna '{hl_col}' não existe no DataFrame.")

        col_idx = df.columns.get_loc(hl_col)
        x_start = col_idx
        x_end = col_idx + 1
        y_start = 0
        y_end = df.shape[0]

        kwargs = {
            'color' : "yellow", 
            "linewidth": 3
        }

        ax.add_line(plt.Line2D([x_start, x_start], [y_start, y_end], **kwargs))  
        ax.add_line(plt.Line2D([x_end, x_end], [y_start, y_end], **kwargs))      
        ax.add_line(plt.Line2D([x_start, x_end], [y_start, y_start], **kwargs))  
        ax.add_line(plt.Line2D([x_start, x_end], [y_end, y_end], **kwargs))      

    for row_idx, (_, row) in enumerate(df.iterrows()):
        min_col_idx = row.idxmin()
        col_idx = df.columns.get_loc(min_col_idx)
        ax.text(
            x=col_idx + 0.99, 
            y=row_idx - 0.05, 
            s=f"min", 
            ha='right', 
            va='top', 
            color='red', 
            fontsize=6.5
        )

    plt.show()
