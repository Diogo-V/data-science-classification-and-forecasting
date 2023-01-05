from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split

from profiling.profiling import Profiler
from preparation.scaling import Scaling

from transformation.aggregation import Aggregation
from transformation.differentiation import Differentiation
from transformation.smoothing import Smoothing

INPUT_FILE_PATH = 'climate-forecasting/resources/data/drought.forecasting_dataset.csv'

if __name__ == "__main__":

  # ----------------------------- 1º Phase -> Data profiling ----------------------------- #
  data = read_csv(INPUT_FILE_PATH, index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst= True)

  multivariate_data = data

  # Remove variables except target
  data = data.drop(columns=['PRECTOT'])	
  data = data.drop(columns=['PS'])	
  data = data.drop(columns=['T2M'])	
  data = data.drop(columns=['T2MDEW'])	
  data = data.drop(columns=['T2MWET'])	
  data = data.drop(columns=['TS'])	

  profiler = Profiler(data)
  profiler.explore_dimensionality()
  profiler.explore_granularity()
  profiler.explore_distribution_boxplots()
  profiler.explore_distribution_histograms()
  profiler.explore_stationary()
  # profiler.explore_count_data_types()

  # ----------------------------- 2º Phase -> Data preparation ----------------------------- #

  ## As shown by explore_count_data_types, there are no missing values for this dataset
  
  # scaling = Scaling(data)
  # data = scaling.compute_scale()

  # differentiation = Differentiation(data)
  # data = differentiation.compute_differentiation()

  aggregation = Aggregation(data)
  aggregation.explore_aggregation()
  data = aggregation.compute_aggregation()

  smoothing = Smoothing(data)
  data = smoothing.explore_smoothing()

  differentiation = Differentiation(data)
  differentiation.explore_differentiation()
  data = differentiation.compute_differentiation()

