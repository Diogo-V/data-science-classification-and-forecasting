import pandas as pd
from numpy import log
from seaborn import distplot
from matplotlib.pyplot import figure, savefig, subplots, Axes
from scipy.stats import norm, expon, lognorm

from ds_charts import bar_chart, get_variable_types, HEIGHT, choose_grid, multiple_bar_chart, multiple_line_chart


class Distribution:

  def __init__(self, data: pd.DataFrame) -> None:
    self.data: pd.DataFrame = data
    data['fips'] = data['fips'].astype('category')
    data['SQ1'] = data['SQ1'].astype('category')
    data['SQ2'] = data['SQ2'].astype('category')
    data['SQ3'] = data['SQ3'].astype('category')
    data['SQ4'] = data['SQ4'].astype('category')
    data['SQ5'] = data['SQ5'].astype('category')
    data['SQ6'] = data['SQ6'].astype('category')
    data['SQ7'] = data['SQ7'].astype('category')
    data['date'] = data['date'].astype('datetime64')

  def explore_global_boxplot(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and saves an images with a boxplot for each variables.

    Arguments:
      * output_image_path(str): path to output image
    """
    self.data.boxplot(rot=45)
    savefig(f'{output_image_path}/global_boxplot.png')

  def explore_numeric_boxplot(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and saves an image with a boxplot for numeric variables.

    Arguments:
      * output_image_path(str): path to output image
    """
    numeric_vars = get_variable_types(self.data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(self.data[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{output_image_path}/numeric_boxplot.png')


  def explore_count_numeric_outliers(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and saves an image with the count of outliers per numeric variable.

    Arguments:
      * output_image_path(str): path to output image
    """
    NR_STDEV: int = 2

    numeric_vars = get_variable_types(self.data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    outliers_iqr = []
    outliers_stdev = []
    summary5 = self.data.describe(include='number')

    for var in numeric_vars:
        iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
        outliers_iqr += [
            self.data[self.data[var] > summary5[var]['75%']  + iqr].count()[var] +
            self.data[self.data[var] < summary5[var]['25%']  - iqr].count()[var]]
        std = NR_STDEV * summary5[var]['std']
        outliers_stdev += [
            self.data[self.data[var] > summary5[var]['mean'] + std].count()[var] +
            self.data[self.data[var] < summary5[var]['mean'] - std].count()[var]]

    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
    savefig(f'{output_image_path}/count_numeric_outliers.png')

  def explore_histogram_numeric_outliers(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and saves a set of histogram images for the numeric variables outliers and their values.

    Arguments:
      * output_image_path(str): path to output image
    """
    numeric_vars = get_variable_types(self.data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(self.data[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{output_image_path}/single_histograms_numeric.png')

  def explore_trend_numeric(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and saves a set of histogram images for the numeric variables with their trend.

    Arguments:
      * output_image_path(str): path to output image
    """
    numeric_vars = get_variable_types(self.data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
        distplot(self.data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{output_image_path}/histograms_trend_numeric.png')

  def explore_symbolic_histogram(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and saves a set of histogram images for the symbolic variables.

    Arguments:
      * output_image_path(str): path to output image
    """
    symbolic_vars = get_variable_types(self.data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')
    
    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT * 2, rows*HEIGHT * 2), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = self.data[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{output_image_path}/histograms_symbolic.png')

  def explore_numeric_distributions(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and saves a set of histogram images for the numeric distributions.

    Arguments:
      * output_image_path(str): path to output image
    """
    def compute_known_distributions(x_values: list) -> dict:
      distributions = dict()
      # Gaussian
      mean, sigma = norm.fit(x_values)
      distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = norm.pdf(x_values, mean, sigma)
      # Exponential
      loc, scale = expon.fit(x_values)
      distributions['Exp(%.2f)'%(1/scale)] = expon.pdf(x_values, loc, scale)
      # LogNorm
      sigma, loc, scale = lognorm.fit(x_values)
      distributions['LogNor(%.1f,%.2f)'%(log(scale),sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
      return distributions

    def histogram_with_distributions(ax: Axes, series: pd.Series, var: str):
      values = series.sort_values().values
      ax.hist(values, 20, density=True)
      distributions = compute_known_distributions(values)
      multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')

    res = get_variable_types(self.data)['Numeric']
    k = 0  # Literally had to move it one by one
    numeric_vars = [res[k]]
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        histogram_with_distributions(axs[i, j], self.data[numeric_vars[n]].dropna(), numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{output_image_path}/numeric_distribution/numeric_distribution_{k}.png')