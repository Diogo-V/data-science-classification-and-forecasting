from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from profiling import Profiler

register_matplotlib_converters()

RECORDS_PATH = 'health/records'
FILE_PATH = 'health/resources/data/diabetic_data.csv'

if __name__ == "__main__":
  data = read_csv(FILE_PATH, na_values='na')

  print(data["diag_1"])

  for val in data["diag_1"]:
    if type(val) is str:
      print(val)

  # 1º Phase -> Data profiling
  # profiler = Profiler(data)
  # profiler.explore_dimensionality(RECORDS_PATH)
  # profiler.explore_data_types()
  # profiler.explore_missing_values(RECORDS_PATH, True)
