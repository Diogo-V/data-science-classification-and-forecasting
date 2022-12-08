from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from numpy import nan

from profiling.profiling import Profiler
from profiling.sparsity import Sparsity
from profiling.distribution import Distribution

from preparation.parsing import Parser
from preparation.mvi_inputation import MVImputation
from preparation.outliers_imputation import OutliersImputation
from preparation.scaling import Scaling
from preparation.balancing import Balancing

register_matplotlib_converters()

RECORDS_PATH = 'health/records'
PROFILING_PATH = RECORDS_PATH + '/profiling'
PREPARATION_PATH = RECORDS_PATH + '/preparation'
INPUTATION_PATH = RECORDS_PATH + '/inputation'

INPUT_FILE_PATH = 'health/resources/data/diabetic_data.csv'
PREPARATION_OUT_FILE_PATH = 'health/resources/data/data_prepared.csv'
MVI_OUT_FILE_PATH = 'health/resources/data/data_mvi_approach1.csv'
INPUTATION_OUT_FILE_PATH = 'health/resources/data'
MISSING_VALUES_REPR = '?'


if __name__ == "__main__":

  # ----------------------------- 1º Phase -> Data profiling ----------------------------- #
  
  # profiler = Profiler(data)
  # profiler_sparsity = Sparsity(data)
  # profiler_distribution = Distribution(data)

  # profiler.explore_dimensionality(PROFILING_PATH, display=False)
  # profiler.explore_variable_types(PROFILING_PATH, display=False)
  # profiler.explore_missing_values(PROFILING_PATH, MISSING_VALUES_REPR, display=True)

  # profiler_distribution.explore_global_boxplot(PROFILING_PATH)
  # profiler_distribution.explore_numeric_boxplot(PROFILING_PATH)
  # profiler_distribution.explore_count_numeric_outliers(PROFILING_PATH)
  # profiler_distribution.explore_histogram_numeric_outliers(PROFILING_PATH)
  # profiler_distribution.explore_trend_numeric(PROFILING_PATH)
  # profiler_distribution.explore_symbolic_histogram(PROFILING_PATH)
  # profiler_distribution.explore_numeric_distributions(PROFILING_PATH)

  # profiler.explore_data_granularity(PROFILING_PATH, True, data_type='Numeric')
  # profiler.explore_data_granularity(PROFILING_PATH, True, data_type="Symbolic")
  # profiler.explore_data_granularity(PROFILING_PATH, True, data_type='Date')

  # profiler_sparsity.explore_scatter_plot(PROFILING_PATH)
  # profiler_sparsity.explore_heatmap(PROFILING_PATH)

  # ----------------------------- 2º Phase -> Data preparation -------  ---------------------- #

  # data = read_csv(INPUT_FILE_PATH, na_values='na')
 
  # parser = Parser(data, MISSING_VALUES_REPR)
  # data = parser.parse_dataset(PREPARATION_OUT_FILE_PATH)
 
  # mvi = MVImputation(data, MISSING_VALUES_REPR)
  # data = mvi.compute_mv_imputation()

  # outliers = OutliersImputation(data)
  # data = outliers.compute_outliers()
 
  # scaling = Scaling(data)
  # data = scaling.compute_scale()
 
  # # Removes single value columns
  # ms = [
  #   'repaglinide', 'max_glu_serum', 'nateglinide', 'chlorpropamide', 'acetohexamide', 'acarbose', 'miglitol', 
  #   'tolazamide', 'citoglipton', 'examide', 'glyburide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone'
  # ]
  # data = data.drop(columns=ms)
  #  
  # balancing = Balancing(data)
  # data = balancing.compute_balancing()
  # 
  # data.to_csv(PREPARATION_OUT_FILE_PATH)

  # ----------------------------- 3º Phase -> Evaluation -------  ---------------------- #

  data = read_csv(PREPARATION_OUT_FILE_PATH, na_values='na')
