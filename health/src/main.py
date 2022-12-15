from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
import numpy as np

from profiling.profiling import Profiler
from profiling.sparsity import Sparsity
from profiling.distribution import Distribution

from preparation.parsing import Parser
from preparation.mvi_inputation import MVImputation
from preparation.outliers_imputation import OutliersImputation
from preparation.scaling import Scaling
from preparation.balancing import Balancing
from evaluation.nb_classifier import NBClassifier
from evaluation.knn_classifier import Knn_classifier

register_matplotlib_converters()

RECORDS_PATH = 'health/records'
PROFILING_PATH = RECORDS_PATH + '/profiling'
PREPARATION_PATH = RECORDS_PATH + '/preparation'
INPUTATION_PATH = RECORDS_PATH + '/inputation'

INPUT_FILE_PATH = 'health/resources/data/diabetic_data.csv'
PREPARATION_OUT_FILE_PATH = 'health/resources/data/data_prepared.csv'
PREPARATION_OUT_FILE_PATH_TRAIN = 'health/resources/data/data_prepared_train.csv'
PREPARATION_OUT_FILE_PATH_TEST = 'health/resources/data/data_prepared_test.csv'
MVI_OUT_FILE_PATH = 'health/resources/data/data_mvi_approach2.csv'
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
  # data = mvi.compute_mv_imputation(INPUTATION_OUT_FILE_PATH)

  # outliers = OutliersImputation(data)
  # data = outliers.compute_outliers()
  
  # scaling = Scaling(data)
  # data = scaling.compute_scale()
  
  # # Removes single value columns
  # ms = [
  #   'acetohexamide', 'examide', 'citoglipton', 
  # ]
  # data = data.drop(columns=ms)

  # # Splits data before evaluation
  # X = data.drop("readmitted", axis=1)
  # y = data["readmitted"]
  # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

  # data_train = concat([DataFrame(X_train), DataFrame(y_train)], axis=1)

  # balancing = Balancing(data_train)
  # data = balancing.compute_balancing()

  # data.to_csv(PREPARATION_OUT_FILE_PATH_TRAIN)
  
  # data_test = concat([DataFrame(X_test), DataFrame(y_test)], axis=1)
  # data_test.to_csv(PREPARATION_OUT_FILE_PATH_TEST)

  # ----------------------------- 3º Phase -> Evaluation -------  ---------------------- #

  data_train = read_csv(PREPARATION_OUT_FILE_PATH_TRAIN, na_values='na')
  data_test = read_csv(PREPARATION_OUT_FILE_PATH_TEST, na_values='na')

  # nbClassifier = NBClassifier(data_train, data_test)
  # nbClassifier.explore_best_nb_value()
  # nbClassifier.compute_nb_best_results()

  knn_class = Knn_classifier(data_train, data_test)
  k, approach = knn_class.explore_best_k_value(method="large")
  # knn_class.compute_knn_best_results(k , approach)
