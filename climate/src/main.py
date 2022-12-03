from pandas import read_csv
from pandas.plotting import register_matplotlib_converters

from profiling.profiling import Profiler
from profiling.sparsity import Sparsity
from profiling.distribution import Distribution

from preparation.scaling import Scaling

register_matplotlib_converters()

RECORDS_PATH = 'climate/records'
PROFILING_PATH = RECORDS_PATH + '/profiling'
PREPARATION_DATA = RECORDS_PATH + '/preparation'
FILE_PATH = 'climate/resources/data/drought.csv'

if __name__ == "__main__":
  data = read_csv(FILE_PATH, na_values='na')

  profiler = Profiler(data)
  profiler_distribution = Distribution(data)

  scaling = Scaling(data)
  scaling.analyze_scaling()

  # profiler.explore_data_granularity(PROFILING_PATH, True, data_type='Numeric')
  # profiler.explore_data_granularity(PROFILING_PATH, True, data_type="Symbolic")
  # profiler.explore_data_granularity(PROFILING_PATH, True, data_type='Date')

  # profiler_sparsity = Sparsity(data)
  # profiler_sparsity.explore_scatter_plot(PROFILING_PATH)
  # profiler_sparsity.explore_heatmap(PROFILING_PATH)

  # profiler_distribution.explore_global_boxplot(PROFILING_PATH)
  # profiler_distribution.explore_numeric_boxplot(PROFILING_PATH)
  # profiler_distribution.explore_count_numeric_outliers(PROFILING_PATH)
  # profiler_distribution.explore_histogram_numeric_outliers(PROFILING_PATH)
  # profiler_distribution.explore_trend_numeric(PROFILING_PATH)
  # profiler_distribution.explore_symbolic_histogram(PROFILING_PATH)
  # profiler_distribution.explore_numeric_distributions(PROFILING_PATH)