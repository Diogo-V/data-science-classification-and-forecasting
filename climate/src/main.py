from pandas import read_csv, DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots
from ds_charts import bar_chart, get_variable_types, choose_grid, HEIGHT, multiple_bar_chart
from seaborn import distplot

NR_STDEV: int = 2

register_matplotlib_converters()
# filename = 'data/algae.csv'
# data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

filename = 'climate/resources/data/drought.csv'
data = read_csv(filename, na_values='na')

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
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
        elif df[c].dtype == 'datetime64':
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
  savefig('climate/images/records_variables.png')
  show()

def nr_of_variables_per_type():
  variable_types = get_variable_types(data)
  print(variable_types)
  counts = {}
  for tp in variable_types.keys():
      counts[tp] = len(variable_types[tp])
  figure(figsize=(4,2))
  bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
  savefig('climate/images/variable_types.png')
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
  savefig('climate/images/mv.png')
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
  savefig('climate/images/single_boxplots.png')
  show()

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
  multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
  savefig('climate/images/outliers.png')
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
  savefig('climate/images/single_histograms_numeric.png')
  show()


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
  savefig('climate/images/histograms_trend_numeric.png')
  show()


if __name__ == "__main__":
  histogram_trend_per_numeric_variable()