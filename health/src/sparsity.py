import pandas as pd
from matplotlib.pyplot import subplots, savefig, title, figure
from ds_charts import HEIGHT
from seaborn import heatmap


class Sparsity:

  def __init__(self, data: pd.DataFrame) -> None:
    self.data: pd.DataFrame = data

  def explore_scatter_plot(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and optionally displays a set of graphs that shows the matrix of scatter plots.

    Arguments:
      * output_image_path(str): path to output image
    """
    columns = list(self.data)
    rows, cols = len(columns)-1, len(columns)-1
    _, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(columns)):
        var1 = columns[i]
        for j in range(i+1, len(columns)):
            var2 = columns[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(self.data[var1], self.data[var2])
    savefig(f'{output_image_path}/sparsity_study.png')

  def explore_heatmap(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and optionally displays the correlation matrix.

    Arguments:
      * output_image_path(str): path to output image
    """
    figure(figsize=[12, 12])
    corr_mtx = abs(self.data.corr())
    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation analysis')
    savefig(f'{output_image_path}/correlation_analysis.png')