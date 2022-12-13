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

from evaluation.knn_classifier import Knn_classifier
from evaluation.nb_classifier import NBClassifier

register_matplotlib_converters()

RECORDS_PATH = 'climate/records'
PREPARATION_OUT_FILE_PATH = 'climate/resources/data/data_prepared.csv'
PREPARATION_OUT_FILE_PATH_TRAIN = 'climate/resources/data/data_prepared_train.csv'
PREPARATION_OUT_FILE_PATH_TEST = 'climate/resources/data/data_prepared_test.csv'
PROFILING_PATH = RECORDS_PATH + '/profiling'
PREPARATION_DATA = RECORDS_PATH + '/preparation'
FILE_PATH = 'climate/resources/data/drought.csv'


if __name__ == "__main__":

  # ----------------------------- 2º Phase -> Data preparation -------  ---------------------- #

  # data = read_csv(FILE_PATH, na_values='na')

  # parser = Parser(data)
  # data = parser.parse_dataset(PREPARATION_OUT_FILE_PATH)

  # outliers = OutliersImputation(data)
  # data = outliers.compute_outliers()
  
  # scaling = Scaling(data)
  # data = scaling.compute_scale()

  # # Remove single values columns
  # ms = ['NVG_LAND', 'SQ5', 'SQ6']
  # data = data.drop(columns=ms)

  # # Splits data before evaluation
  # X = data.drop("class", axis=1)
  # y = data["class"]
  # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

  # data_train = concat([DataFrame(X_train), DataFrame(y_train)], axis=1)

  # balancing = Balancing(data_train)
  # data = balancing.compute_balancing()
  # data.to_csv(PREPARATION_OUT_FILE_PATH_TRAIN)
  
  # data_test = concat([DataFrame(X_test), DataFrame(y_test)], axis=1)
  # data_test.to_csv(PREPARATION_OUT_FILE_PATH_TEST)

  # # ----------------------------- 3º Phase -> Evaluation -------  ---------------------- #

  data_train = read_csv(PREPARATION_OUT_FILE_PATH_TRAIN, na_values='na')
  data_test = read_csv(PREPARATION_OUT_FILE_PATH_TEST, na_values='na')

  nbClassifier = NBClassifier(data_train, data_test)
  nbClassifier.explore_best_nb_value()
  # nbClassifier.compute_nb_best_results()

  # knn_class = Knn_classifier(data_train, data_test)
  # k, approach = knn_class.explore_best_k_value()
  # knn_class.compute_knn_best_results(k , approach)
