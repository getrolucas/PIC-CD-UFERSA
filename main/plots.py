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
        figsize: tuple = (6, 6),
        annot_kws : dict = {"fontsize": 9},
        low_is_better : bool = True,
        title : str = 'Performance dos Modelos Para Cada Série Temporal',
        caption : str = 'Métrica de desempenho: Raiz do Erro Quadrático Médio.'
    ) -> None:
    
    """
    Plota as métricas de desempenho dos modelos por série em um heatmap 
    possibilitando comparar visualmente qual modelo foi melhor.

    Args:
        df (pd.DataFrame): DataFrame com os dados a serem plotados.
        figsize (tuple): Tamanho da figura.
        annot_kws (dict): Kwargs para ajuste da fonte da anotação.
        low_is_better (bool): Define se a métrica é melhor quando o valor é menor.
        title (str): Título do gráfico.
        caption (str): Legenda do gráfico.
    """
    
    title_fontsize = 12
    cmap = 'Greens'
    fontdict = {'weight': 'bold'}
    title_fontdict = {'size': title_fontsize, 'weight':'bold'}
    
    # estrutura básica do gráfico com dados escalonados entre 0 e 1
    fig = plt.figure(figsize=figsize)
    
    ax = sns.heatmap(
        data=df.T.apply(_minmax),
        fmt=".2f",
        cbar=False,
        cmap=cmap + '_r' if low_is_better else cmap,
        annot_kws=annot_kws,
        # vmin=-0.25 if low_is_better else 0, 
        # vmax=1 if low_is_better else 1.25,
    )
    
    
    # anotando os maps com valores originais
    # destacando os valores mínimos ou máximos
    for i, (_, row) in enumerate(df.iterrows()):
        best_col_idx = row.idxmin() if low_is_better else row.idxmax()
        best_val = row[best_col_idx]
        for ii, _ in enumerate(df.columns):
            ax.text(
                x=i + 0.5, 
                y=ii + 0.5, 
                s=f"{row.iloc[ii]:.2f}", 
                ha='center',
                va='center',
                fontsize=annot_kws['fontsize'] + 0.75,
                color='white' if row.iloc[ii] == best_val else 'black',
                fontdict=fontdict if row.iloc[ii] == best_val else None
            )


    # configs do eixo X
    plt.gca().xaxis.tick_top()
    plt.xticks(fontsize=annot_kws['fontsize'], **fontdict)
    plt.tick_params(axis='x', length=0)
    plt.xlabel('')


    # configs do eixo Y
    plt.yticks(fontsize=annot_kws['fontsize'])
    plt.tick_params(axis='y', length=0)
    plt.ylabel('')
    
    
    # configs das linhas nas bordas do heatmap
    ax.axvline(color='black', linewidth=1, clip_on=False, x=0, ymax=1) 
    ax.axhline(color='black', linewidth=1, clip_on=False, y=0, xmin=0)    
    
    # configs do título do gráfico
    plt.title(label=title, fontdict=title_fontdict , pad=30)
    
    
    ## configs da legenda do gráfico
    plt.figtext(
        x=0.9, y=-0.0025, horizontalalignment='right', fontsize=annot_kws['fontsize'],
        s=caption, fontdict=fontdict
    )
    
    return ax.get_xticklabels()
