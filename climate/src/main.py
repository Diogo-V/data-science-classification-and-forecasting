from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split

from profiling.profiling import Profiler
from profiling.sparsity import Sparsity
from profiling.distribution import Distribution

from preparation.parsing import Parser
from preparation.scaling import Scaling
from preparation.outliers_imputation import OutliersImputation
from preparation.balancing import Balancing
from preparation.feature_selection import FeatureSelection

from evaluation.knn_classifier import Knn_classifier
from evaluation.nb_classifier import NBClassifier
from evaluation.random_forest_classifier import RTClassifier
from evaluation.decision_tree_classifier import DTClassifier

register_matplotlib_converters()

RECORDS_PATH = 'climate/records'
PREPARATION_OUT_FILE_PATH = 'climate/resources/data/data_prepared.csv'
FEATURE_SELECTION_OUT_FILE_PATH_TRAIN = 'climate/resources/data/data_feature_selected.csv'
PREPARATION_OUT_FILE_PATH_TRAIN = 'climate/resources/data/data_prepared_train.csv'
PREPARATION_OUT_FILE_PATH_TEST = 'climate/resources/data/data_prepared_test.csv'
PROFILING_PATH = RECORDS_PATH + '/profiling'
PREPARATION_DATA = RECORDS_PATH + '/preparation'
FILE_PATH = 'climate/resources/data/drought.csv'


if __name__ == "__main__":

  # ----------------------------- 2º Phase -> Data preparation -------  ---------------------- #

  """  data = read_csv(FILE_PATH, na_values='na')

  parser = Parser(data)
  data = parser.parse_dataset(PREPARATION_OUT_FILE_PATH)

  outliers = OutliersImputation(data)
  data = outliers.compute_outliers()
  
  scaling = Scaling(data)
  data = scaling.compute_scale()

  # Feature Selection
  feature_Selection = FeatureSelection(data)
  vars_2drop = feature_Selection.explore_redundat()
  data = feature_Selection.drop_redundant(vars_2drop)
  data.to_csv(FEATURE_SELECTION_OUT_FILE_PATH_TRAIN)
  feature_Selection.select_low_variance(data)
  
  # Remove single values columns
  ms = ['NVG_LAND', 'SQ5', 'SQ6']
  data = data.drop(columns=ms)

  # Splits data before evaluation
  X = data.drop("class", axis=1)
  y = data["class"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

  data_train = concat([DataFrame(X_train), DataFrame(y_train)], axis=1)

  balancing = Balancing(data_train)
  data = balancing.compute_balancing()
  data.to_csv(PREPARATION_OUT_FILE_PATH_TRAIN, index=False)
  
  data_test = concat([DataFrame(X_test), DataFrame(y_test)], axis=1)
  data_test.to_csv(PREPARATION_OUT_FILE_PATH_TEST, index=False) """

  # # ----------------------------- 3º Phase -> Evaluation -------  ---------------------- #

  data_train = read_csv(PREPARATION_OUT_FILE_PATH_TRAIN, na_values='na')
  data_test = read_csv(PREPARATION_OUT_FILE_PATH_TEST, na_values='na')

  # nbClassifier = NBClassifier(data_train, data_test)
  # nbClassifier.explore_best_nb_value()
  # nbClassifier.compute_nb_best_results()

  # knn_class = Knn_classifier(data_train, data_test)
  # k, approach = knn_class.explore_best_k_value(method="large")
  # knn_class.compute_knn_best_results(k , approach)

  rtClassifier = RTClassifier(data_train, data_test)
  # rtClassifier.explore_best_rt()
  rtClassifier.compute_best_rt_results(25, 0.7, 400)

  # dt_classifier = DTClassifier(data_train, data_test)
  # criteria, depth, impurity = dt_classifier.compute_best_dt()
  # dt_classifier.explore_best_tree_graph_light()
  # dt_classifier.explore_dt_best_matrix_results(depth, criteria, impurity)
  # dt_classifier.explore_dt_feature_importance()
  # dt_classifier.explore_best_dt_overfit(criteria, impurity)
