from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from ts_functions import split_dataframe

from profiling.profiling import Profiler
from preparation.mvi_imputation import MVImputation
from preparation.scaling import Scaling

from transformation.aggregation import Aggregation
from transformation.differentiation import Differentiation
from transformation.smoothing import Smoothing

from evaluation.arima import ARIMA

INPUT_FILE_PATH = 'health-forecasting/resources/data/glucose.csv'

if __name__ == "__main__":

    # ----------------------------- 1º Phase -> Data profiling ----------------------------- #
    # data = read_csv(INPUT_FILE_PATH, index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

    # multivariate_data = data

    # Remove variables except target
    # data = data.drop(columns=['Insulin'])	

    # profiler = Profiler(data)
    # profiler.explore_dimensionality()
    # profiler.explore_granularity()
    # profiler.explore_distribution_boxplots()
    # profiler.explore_distribution_histograms()
    # profiler.explore_stationary()

    # ----------------------------- 2º Phase -> Data preparation ----------------------------- #
    # mvi = MVImputation(data)
    # mvi.explore_mv_imputation()
    # data = mvi.compute_mv_imputation("approach_2")

    # No scaling applied due to few columns in dataset
    
    # aggregation = Aggregation(data)
    # aggregation.explore_aggregation()
    # data = aggregation.compute_aggregation()

    # smoothing = Smoothing(data)
    # smoothing.explore_smoothing()
    # data = smoothing.compute_smoothing()

    # differentiation = Differentiation(data)
    # differentiation.explore_differentiation()
    # data = differentiation.compute_differentiation()

    # ----------------------------- 3º Phase -> Data evaluation ----------------------------- #
    data = read_csv(INPUT_FILE_PATH, index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
    data = data.drop(columns=['Insulin'])	
    train, test = split_dataframe(data, trn_pct=0.75)

    arima = ARIMA(train)
    arima.explore_arima(test)
