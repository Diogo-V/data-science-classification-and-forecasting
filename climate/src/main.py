from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from profiling.profiling import Profiler
from profiling.sparsity import Sparsity
from profiling.distribution import Distribution

from preparation.scaling import Scaling
from preparation.outliers_imputation import OutliersImputation
from preparation.balancing import Balancing

register_matplotlib_converters()

RECORDS_PATH = 'climate/records'
PREPARATION_OUT_FILE_PATH = 'climate/resources/data/data_prepared.csv'
PROFILING_PATH = RECORDS_PATH + '/profiling'
PREPARATION_DATA = RECORDS_PATH + '/preparation'
FILE_PATH = 'climate/resources/data/drought.csv'


if __name__ == "__main__":

  # ----------------------------- 2º Phase -> Data preparation -------  ---------------------- #

  data = read_csv(FILE_PATH, na_values='na')

  outliers = OutliersImputation(data)
  data = outliers.compute_outliers()
  
  scaling = Scaling(data)
  data = scaling.compute_scale()

  # Remove single values columns
  ms = ['NVG_LAND', 'SQ5', 'SQ6']
  data = data.drop(columns=ms)

  balancing = Balancing(data)
  data = balancing.compute_balancing()

  profiler = Profiler(data)
  profiler.count_unique()

  # data.to_csv(PREPARATION_OUT_FILE_PATH) 

  # ----------------------------- 3º Phase -> Evaluation -------  ---------------------- #

  # data = read_csv(PREPARATION_OUT_FILE_PATH, na_values='na')
