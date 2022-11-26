import pandas as pd
from matplotlib.pyplot import figure, savefig, show

from ds_charts import bar_chart


class Profiler:

  def __init__(self, data: pd.DataFrame) -> None:
    self.data: pd.DataFrame = data

  def explore_dimensionality(self, output_image_path: str, display=False) -> None:
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

  def explore_data_types(self) -> None:
    """
    Description:
      * Shows data types for each column.
    """
    print(self.data.dtypes)

  def explore_missing_values(self, output_image_path: str, display=False) -> None:
    """
    Description:
      * Builds and optionally displays a graph that shows the number of missing values.

    Arguments:
      * output_image_path(str): path to output image
      * display(bool): boolean that controls display or not of built graph
    """
    mv = {}
    for var in self.data:
        nr = self.data[var].isna().sum()
        if nr > 0:
            mv[var] = nr
    
    figure()
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
                xlabel='variables', ylabel='nr missing values', rotation=True)
    savefig(f'{output_image_path}/nr_missing_values.png')
    if display:
        show()