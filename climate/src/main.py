from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from profiling.profiling import Profiler
from profiling.sparsity import Sparsity
from profiling.distribution import Distribution

from preparation.scaling import Scaling
from preparation.outliers_imputation import OutliersImputation

register_matplotlib_converters()

RECORDS_PATH = 'climate/records'
PROFILING_PATH = RECORDS_PATH + '/profiling'
PREPARATION_DATA = RECORDS_PATH + '/preparation'
FILE_PATH = 'climate/resources/data/drought.csv'

if __name__ == "__main__":
  data = read_csv(FILE_PATH, na_values='na')

  outliers = OutliersImputation(data)
  data = outliers.compute_outliers()
  
  scaling = Scaling(data)
  data = scaling.compute_scale()