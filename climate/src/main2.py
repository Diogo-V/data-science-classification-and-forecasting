from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from profiling import Profiler
from sparsity import Sparsity
from distribution import Distribution

register_matplotlib_converters()

RECORDS_PATH = 'climate/records'
FILE_PATH = 'climate/resources/data/drought.csv'

if __name__ == "__main__":
  data = read_csv(FILE_PATH, na_values='na')

  profiler = Profiler(data)
  profiler_distribution = Distribution(data)

  # profiler.explore_data_granularity(RECORDS_PATH, True, data_type='Numeric')
  # profiler.explore_data_granularity(RECORDS_PATH, True, data_type="Symbolic")
  # profiler.explore_data_granularity(RECORDS_PATH, True, data_type='Date')

  # profiler_sparsity = Sparsity(data)
  # profiler_sparsity.explore_scatter_plot(RECORDS_PATH)
  # profiler_sparsity.explore_heatmap(RECORDS_PATH)

  profiler_distribution.explore_global_boxplot(RECORDS_PATH)
  # profiler_distribution.explore_numeric_boxplot(RECORDS_PATH)
  # profiler_distribution.explore_count_numeric_outliers(RECORDS_PATH)
  # profiler_distribution.explore_histogram_numeric_outliers(RECORDS_PATH)
  # profiler_distribution.explore_trend_numeric(RECORDS_PATH)
  # profiler_distribution.explore_symbolic_histogram(RECORDS_PATH)
  # profiler_distribution.explore_numeric_distributions(RECORDS_PATH)