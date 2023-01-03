from pandas import read_csv, DataFrame, concat
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split

from profiling.profiling import Profiler
from preparation.mvi_imputation import MVImputation
from preparation.scaling import Scaling

INPUT_FILE_PATH = 'health-forecasting/resources/data/glucose.csv'

if __name__ == "__main__":

    # ----------------------------- 1º Phase -> Data profiling ----------------------------- #
    data = read_csv(INPUT_FILE_PATH, index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)

    #  profiler = Profiler(data)
    #  profiler.explore_count_data_types()

    # ----------------------------- 2º Phase -> Data preparation ----------------------------- #
    # mvi = MVImputation(data)
    # data = mvi.approach_1()
    # data = mvi.approach_2()

    ## Does it even make sense to apply scaling?
    # scaling = Scaling(data)
    # data = scaling.explore_scaling()
