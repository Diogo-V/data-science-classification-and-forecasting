import pandas as pd
from matplotlib.pyplot import figure, savefig, show, subplots

from ds_charts import bar_chart, get_variable_types, HEIGHT


class Profiler:

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

  def get_variable_types(self, df: pd.DataFrame) -> dict:
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

  def explore_data_granularity(self, output_image_path: str, display=False, data_type: str = 'Numeric') -> None:
    variables = self.get_variable_types(self.data)[data_type]
    bins = (10, 100, 1000)
    rows = len(variables)
    cols = len(bins)
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for i in range(rows):
      for j in range(cols):
        axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
        axs[i, j].set_xlabel(variables[i])
        axs[i, j].set_ylabel('Nr records')
        axs[i, j].hist(self.data[variables[i]].values, bins=bins[j])
    savefig(f'{output_image_path}/granularity_{data_type}.png')
    if display:
      show()
