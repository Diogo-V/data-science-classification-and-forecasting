from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split

from profiling.profiling import Profiler

INPUT_FILE_PATH = 'climate-forecasting/resources/data/drought.forecasting_dataset.csv'

if __name__ == "__main__":

  # ----------------------------- 1º Phase -> Data profiling ----------------------------- #
  data = read_csv(INPUT_FILE_PATH, index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

  profiler = Profiler(data)
#   profiler.explore_dimensionality()
#   profiler.explore_granularity()
#   profiler.explore_distribution_boxplots()
#   profiler.explore_distribution_histograms()
  profiler.explore_count_data_types()

  # ----------------------------- 2º Phase -> Data preparation ----------------------------- #
  
  ## As shown by explore_count_data_types, there are no missing values for this dataset