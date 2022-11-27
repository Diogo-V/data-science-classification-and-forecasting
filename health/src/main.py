from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from profiling import Profiler
from sparsity import Sparsity
from distribution import Distribution

register_matplotlib_converters()

RECORDS_PATH = 'health/records'
FILE_PATH = 'health/resources/data/diabetic_data.csv'
MISSING_VALUES_REPR = '?'

if __name__ == "__main__":
  data = read_csv(FILE_PATH, na_values='na')

  # 1º Phase -> Data profiling
  profiler = Profiler(data)
  profiler_sparsity = Sparsity(data)
  profiler_distribution = Distribution(data)

  # profiler.explore_dimensionality(RECORDS_PATH, display=False)
  # profiler.explore_variable_types(RECORDS_PATH, display=False)
  # profiler.explore_missing_values(RECORDS_PATH, MISSING_VALUES_REPR, display=True)

  # profiler_distribution.explore_global_boxplot(RECORDS_PATH)
  # profiler_distribution.explore_numeric_boxplot(RECORDS_PATH)
  # profiler_distribution.explore_count_numeric_outliers(RECORDS_PATH)
  # profiler_distribution.explore_histogram_numeric_outliers(RECORDS_PATH)
  # profiler_distribution.explore_trend_numeric(RECORDS_PATH)
  # profiler_distribution.explore_symbolic_histogram(RECORDS_PATH)
  profiler_distribution.explore_numeric_distributions(RECORDS_PATH)

  # profiler.explore_data_granularity(RECORDS_PATH, True, data_type='Numeric')
  # profiler.explore_data_granularity(RECORDS_PATH, True, data_type="Symbolic")
  # profiler.explore_data_granularity(RECORDS_PATH, True, data_type='Date')

  # profiler_sparsity.explore_scatter_plot(RECORDS_PATH)
  # profiler_sparsity.explore_heatmap(RECORDS_PATH)
