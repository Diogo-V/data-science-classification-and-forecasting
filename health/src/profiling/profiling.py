import pandas as pd
from matplotlib.pyplot import figure, savefig, show, subplots

from ds_charts import bar_chart, get_variable_types, HEIGHT


class Profiler:

  def __init__(self, data: pd.DataFrame) -> None:
    self.data: pd.DataFrame = data
    data['admission_type_id'] = data['admission_type_id'].astype('category')
    data['discharge_disposition_id'] = data['discharge_disposition_id'].astype('category')
    data['admission_source_id'] = data['admission_source_id'].astype('category')

  def explore_dimensionality(self, output_image_path: str, display: bool = False) -> None:
    """
    Description:
      * Builds and optionally displays a graph that shows the ratio between the number of records and variables.

    Arguments:
      * output_image_path(str): path to output image
      * display(bool): boolean that controls display or not of built graph
    """
    figure(figsize=(4,2))
    values = {'nr records': self.data.shape[0], 'nr variables': self.data.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
    savefig(f'{output_image_path}/dimensionality.png')
    if display:
        show()
  
  def explore_variable_types(self, output_image_path: str, display: bool = False) -> dict:
    """
    Description:
      * Builds and optionally displays a graph that shows the ratio between the number of records and variables.

    Arguments:
      * output_image_path(str): path to output image
      * display(bool): boolean that controls display or not of built graph
    """
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in self.data.columns:
        uniques = self.data[c].dropna(inplace=False).unique()
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            self.data[c].astype('bool')
        elif self.data[c].dtype == 'datetime64':
            variable_types['Date'].append(c)
        elif self.data[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif self.data[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        else:
            self.data[c].astype('category')
            variable_types['Symbolic'].append(c)

    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4,2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    savefig(f'{output_image_path}/variable_types.png')
    if display:
      show()

  def explore_count_data_types(self) -> None:
    """
    Description:
      * Shows counts of variables per column and missing values percentage.
    """
    for col in self.data:
      print(f"Col name: {col}")
      series = self.data[col].value_counts(ascending=True)
      print(series)
      dc = series.to_dict()
      if dc.get('?') is not None:
        print(f"Percentage of missing values: {round(dc['?'] / sum(dc.values()) * 100, 2)}%")
      print("###################################")

  def explore_missing_values(self, output_image_path: str, missing_value_str: str, display: bool = False) -> None:
    """
    Description:
      * Builds and optionally displays a graph that shows the number of missing values.

    Arguments:
      * output_image_path(str): path to output image
      * missing value_str(str): missing value representation
      * display(bool): boolean that controls display or not of built graph
    """
    mv = {}
    for var in self.data:
        nr = self.data[var].value_counts().to_dict().get(missing_value_str)
        if nr is not None and nr > 0:
            mv[var] = nr

    figure(figsize=[10, 5])
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
                xlabel='variables', ylabel='nr missing values', rotation=False)
    savefig(f'{output_image_path}/nr_missing_values.png')
    if display:
        show()


  def explore_data_granularity(self, output_image_path: str, display=False, data_type: str = 'Numeric') -> None:
    variables = get_variable_types(self.data)[data_type]
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
