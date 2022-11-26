from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from profiling import Profiler

register_matplotlib_converters()

RECORDS_PATH = 'health/records'
FILE_PATH = 'health/resources/data/diabetic_data.csv'
MISSING_VALUES_REPR = '?'

if __name__ == "__main__":
  data = read_csv(FILE_PATH, na_values='na')

  # 1º Phase -> Data profiling
  profiler = Profiler(data)
  profiler.explore_dimensionality(RECORDS_PATH, display=False)
  profiler.explore_variable_types(RECORDS_PATH, display=False)
  profiler.explore_missing_values(RECORDS_PATH, MISSING_VALUES_REPR, display=True)
