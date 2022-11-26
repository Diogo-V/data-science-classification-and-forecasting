from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from profiling import Profiler

register_matplotlib_converters()

RECORDS_PATH = 'climate/records'
FILE_PATH = 'climate/resources/data/drought.csv'

if __name__ == "__main__":
  data = read_csv(FILE_PATH, na_values='na')

  profiler = Profiler(data)

  profiler.explore_data_granularity(RECORDS_PATH, True, data_type='Numeric')
  profiler.explore_data_granularity(RECORDS_PATH, True, data_type="Symbolic")
  profiler.explore_data_granularity(RECORDS_PATH, True, data_type='Date')