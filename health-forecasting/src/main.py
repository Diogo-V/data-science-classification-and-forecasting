from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split

from profiling.profiling import Profiler
from preparation.mvi_imputation import MVImputation
from preparation.scaling import Scaling

from transformation.differentiation import Differentiation

INPUT_FILE_PATH = 'health-forecasting/resources/data/glucose.csv'

if __name__ == "__main__":

    # ----------------------------- 1º Phase -> Data profiling ----------------------------- #
    data = read_csv(INPUT_FILE_PATH, index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

    #  profiler = Profiler(data)
    #  profiler.explore_count_data_types()

    # ----------------------------- 2º Phase -> Data preparation ----------------------------- #
    mvi = MVImputation(data)
    mvi.explore_mv_imputation()
    data = mvi.compute_mv_imputation("approach_2")

    ## Does it even make sense to apply scaling?
    # scaling = Scaling(data)
    # data = scaling.explore_scaling()
    
    # TODO: @Carlos -> After solving MVI, remove the following line:
    # data = data.dropna(inplace=False)

    # differentiation = Differentiation(data)
    # data = differentiation.explore_differentiation()
