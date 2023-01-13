from pandas import read_csv, DataFrame, Series
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots, Axes, title
from ds_charts import bar_chart, get_variable_types, choose_grid, HEIGHT, multiple_bar_chart, multiple_line_chart
from seaborn import distplot
from numpy import log
from scipy.stats import norm, expon, lognorm
from seaborn import heatmap


NR_STDEV: int = 2

register_matplotlib_converters()
# filename = 'data/algae.csv'
# data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

filename = 'climate/resources/data/drought.csv'
data = read_csv(filename, na_values='na')

data['SQ1'] = data['SQ1'].astype('category')
data['SQ2'] = data['SQ2'].astype('category')
data['SQ3'] = data['SQ3'].astype('category')
data['SQ4'] = data['SQ4'].astype('category')
data['SQ5'] = data['SQ5'].astype('category')
data['SQ6'] = data['SQ6'].astype('category')
data['SQ7'] = data['SQ7'].astype('category')
data['date'] = data['date'].astype('datetime64')

data.shape

## Data Dimensionality

def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if df[c].dtype == 'int' and len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64' or df[c].dtype == 'datetime64[ns]':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)

    return variable_types

def nr_of_records_vs_nr_of_variables():
  figure(figsize=(4,2))
  values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
  bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
  savefig('climate/records/profiling/records_variables.png')
  show()

def nr_of_variables_per_type():
  variable_types = get_variable_types(data)
  counts = {}
  for tp in variable_types.keys():
      counts[tp] = len(variable_types[tp])
  figure(figsize=(4,2))
  bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
  savefig('climate/records/profiling/variable_types.png')
  show()

def missing_values():
  mv = {}
  for var in data:
      nr = data[var].isna().sum()
      if nr > 0:
          mv[var] = nr

  print(len(mv))
  figure()
  bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
              xlabel='variables', ylabel='nr missing values', rotation=True)
  savefig('climate/records/profiling/mv.png')
  show()

## Data Distribution

def boxplot_per_numeric_variable():
  numeric_vars = get_variable_types(data)['Numeric']
  if [] == numeric_vars:
    raise ValueError('There are no numeric variables.')
  rows, cols = choose_grid(len(numeric_vars))
  fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
  i, j = 0, 0
  for n in range(len(numeric_vars)):
      axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
      axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
      i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
  savefig('climate/records/profiling/single_boxplots.png')
  show()

def global_boxplot():
    figure(figsize=[10, 10])
    data.boxplot(rot=45)
    savefig('climate/records/profiling/global_boxplot.png')

def nr_of_outliers_per_numeric_variable():
  numeric_vars = get_variable_types(data)['Numeric']
  if [] == numeric_vars:
      raise ValueError('There are no numeric variables.')

  outliers_iqr = []
  outliers_stdev = []
  summary5 = data.describe(include='number')

  for var in numeric_vars:
      iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
      outliers_iqr += [
          data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
          data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
      std = NR_STDEV * summary5[var]['std']
      outliers_stdev += [
          data[data[var] > summary5[var]['mean'] + std].count()[var] +
          data[data[var] < summary5[var]['mean'] - std].count()[var]]

  outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
  figure(figsize=(12, HEIGHT))
  multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False, rotation=True)
  savefig('climate/records/profiling/outliers.png')
  show()


def histogram_per_numeric_variable():
  numeric_vars = get_variable_types(data)['Numeric']
  if [] == numeric_vars:
      raise ValueError('There are no numeric variables.')

  rows, cols = choose_grid(len(numeric_vars))
  fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
  i, j = 0, 0
  for n in range(len(numeric_vars)):
      axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
      axs[i, j].set_xlabel(numeric_vars[n])
      axs[i, j].set_ylabel("nr records")
      axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
      i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
  savefig('climate/records/profiling/single_histograms_numeric.png')

def histogram_per_symbolic_variable():
  symbolic_vars = get_variable_types(data)['Symbolic']
  if [] == symbolic_vars:
      raise ValueError('There are no numeric variables.')

  rows, cols = choose_grid(len(symbolic_vars))
  fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
  i, j = 0, 0
  for n in range(len(symbolic_vars)):
      axs[i, j].set_title('Histogram for %s'%symbolic_vars[n])
      axs[i, j].set_xlabel(symbolic_vars[n])
      axs[i, j].set_ylabel("nr records")
      axs[i, j].hist(data[symbolic_vars[n]].dropna().values, 'auto')
      i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
  savefig('climate/records/profiling/single_histograms_symbolic.png')


def histogram_trend_per_numeric_variable():
  numeric_vars = get_variable_types(data)['Numeric']
  if [] == numeric_vars:
      raise ValueError('There are no numeric variables.')
  rows, cols = choose_grid(len(numeric_vars))
  fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
  i, j = 0, 0
  for n in range(len(numeric_vars)):
      axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
      distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
      i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
  savefig('climate/records/profiling/histograms_trend_numeric.png')
  show()

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

def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values = series.sort_values().values
    ax.hist(values, 20, density=True)
    distributions = compute_known_distributions(values)
    multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s'%var, xlabel=var, ylabel='')


def histogram_with_distributions_per_numeric_variable():
  numeric_vars = get_variable_types(data)['Numeric']
  if [] == numeric_vars:
      raise ValueError('There are no numeric variables.')
  rows, cols = choose_grid(len(numeric_vars))
  fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
  i, j = 0, 0
#   for n in range(len(numeric_vars)):
  histogram_with_distributions(axs[i, j], data[numeric_vars[0]].dropna(), numeric_vars[0])
  i, j = (i + 1, 0) if (0+1) % cols == 0 else (i, j + 1)
  savefig('climate/records/profiling/histogram_numeric_distribution_zero.png')
  show()

def explore_scatter_plot(self, output_image_path: str) -> None:
    """
    Description:
      * Builds and optionally displays a set of graphs that shows the matrix of scatter plots.

    Arguments:
      * output_image_path(str): path to output image
    """
    columns = list(self.data)
    rows, cols = 10, len(columns)-1
    _, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(40, 50):
        var1 = columns[i]
        for j in range(i+1, len(columns)):
            var2 = columns[j]
            axs[i - 40, j-1].set_title("%s x %s"%(var1,var2))
            axs[i - 40, j-1].set_xlabel(var1)
            axs[i - 40, j-1].set_ylabel(var2)
            axs[i - 40, j-1].scatter(self.data[var1], self.data[var2])
    savefig('climate/records/profiling/sparsity_study_4.png')

def explore_heatmap() -> None:
    """
    Description:
        * Builds and optionally displays the correlation matrix.

    Arguments:
        * output_image_path(str): path to output image
    """
    corr_mtx = abs(data.corr())
    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation analysis')
    savefig('climate/records/profiling/correlation_analysis.png')


if __name__ == "__main__":
    histogram_with_distributions_per_numeric_variable()
    # histogram_trend_per_numeric_variable()
#   nr_of_outliers_per_numeric_variable()
    # boxplot_per_numeric_variable()
    # global_boxplot()
    # histogram_per_numeric_variable()
    # histogram_per_symbolic_variable()
    # explore_heatmap()
    