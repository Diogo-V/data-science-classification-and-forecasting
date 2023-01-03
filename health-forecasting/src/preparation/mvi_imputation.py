import pandas as pd
from sklearn.impute import SimpleImputer
from ds_charts import plot_confusion_matrix, plot_evaluation_results_2
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, plot_forecasting_series, HEIGHT

class MVImputation:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Description:
        * Class to transform the original data, dealing with missing values

        Arguments:
        * data(pd.DataFrame): data to to be transformed
        * missing_values_str(str): representation of missing values in dataset
        """

        self.data: pd.DataFrame = data

    def explore_mv_imputation(self):
        self.approach_1()
        self.data = pd.read_csv('health-forecasting/resources/data/glucose.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)
        self.approach_2()
        self.data = pd.read_csv('health-forecasting/resources/data/glucose.csv', index_col='Date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, dayfirst=True)

    def compute_mv_imputation(self, approach) -> pd.DataFrame:
        if approach == "1":
            return self.approach_1_compute()
        else:
            return self.approach_2_compute()

    def approach_1(self):
        """
        - Substitute missing values with mean value of Insulin Column

        - Evaluate with Simple Average
        - Evaluate with Rolling Mean
        """

        imputer = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
        temp_df = pd.DataFrame(imputer.fit_transform(self.data), columns=["Insulin", "Glucose"])
        temp_df.index = self.data.index
        self.data = temp_df
        self.simple_average("approach_1")
        self.rolling_mean("approach_1")

    def approach_2(self):
        """
        - Drop all records with missing values

        - Evaluate with Simple Average
        - Evaluate with Rolling Mean
        """

        self.drop_records()
        self.simple_average("approach_2")
        self.rolling_mean("approach_2")

    def approach_1_compute(self) -> pd.DataFrame:
        """
        - Drop all records with missing values

        - Evaluate with Simple Average
        - Evaluate with Rolling Mean
        """
        imputer = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
        temp_df = pd.DataFrame(imputer.fit_transform(self.data), columns=["Insulin", "Glucose"])
        temp_df.index = self.data.index

        self.data = temp_df
        #self.data.to_csv(f'health-forecasting/resources/data/data_mvi_approach1.csv', index=True)

    def approach_2_compute(self) -> pd.DataFrame:
        """
        - Drop all records with missing values

        - Evaluate with Simple Average
        - Evaluate with Rolling Mean
        """

        self.drop_records()
        #self.data.to_csv(f'health-forecasting/resources/data/data_mvi_approach2.csv', index=True)

    def drop_column(self, column_name: str):
        self.data = self.data.drop(columns=[column_name])	

    def fill_missing_values(self, column_name: str, strategy: str):
        self.data[column_name] = self.data[column_name].apply(lambda x: 0 if type(x) is not str else 1)

    def drop_records(self):
        self.data.dropna(axis=0, how='any', inplace=True)

    def split_dataframe(self, trn_pct=0.70):
        trn_size = int(len(self.data) * trn_pct)
        df_cp =self.data.copy()
        train: DataFrame = df_cp.iloc[:trn_size, :]
        test: DataFrame = df_cp.iloc[trn_size:]
        return train, test

    def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, figname: str, x_label: str = 'time', y_label:str =''):
        _, ax = plt.subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(figname)
        ax.plot(trn.index, trn, label='train', color='b')
        ax.plot(trn.index, prd_trn, '--y', label='train prediction')
        ax.plot(tst.index, tst, label='test', color='g')
        ax.plot(tst.index, prd_tst, '--r', label='test prediction')
        ax.legend(prop={'size': 5})

    def simple_average(self, approach):

        train, test = self.split_dataframe(trn_pct=0.75)
        eval_results = {}

        measure = "R2"
           
        fr_mod = SimpleAvgRegressor()
        fr_mod.fit(train)
        prd_trn = fr_mod.predict(train)
        prd_tst = fr_mod.predict(test)

        eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
        print(eval_results)

        plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'health-forecasting/records/preparation/{approach}_{measure}_simple_avg_eval')
        plt.savefig(f'health-forecasting/records/preparation/{approach}_{measure}_simple_avg_eval.png')
        self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'health-forecasting/records/preparation/{approach}_{measure}_simple_avg_plots', x_label="Date", y_label="Glucose")
        plt.savefig(f'health-forecasting/records/preparation/{approach}_{measure}_simple_avg_plots.png')

    def rolling_mean(self, approach: str) -> None:

        train, test = self.split_dataframe(trn_pct=0.75)
        eval_results = {}

        fr_mod = RollingMeanRegressor()
        fr_mod.fit(train)
        prd_trn = fr_mod.predict(train)
        prd_tst = fr_mod.predict(test)

        measure = "R2"
        eval_results['RollingMean'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
        print(eval_results)
        
        plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'health-forecasting/records/preparation/{approach}_{measure}_rolling_mean_eval')
        plt.savefig(f'health-forecasting/records/preparation/{approach}_{measure}_rolling_mean_eval.png')
        self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'health-forecasting/records/preparation/{approach}_{measure}_rolling_mean_plots', x_label="Date", y_label="Glucose")
        plt.savefig(f'health-forecasting/records/preparation/{approach}_{measure}_rolling_mean_plots.png')


class SimpleAvgRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean = 0

    def fit(self, X: pd.DataFrame):
        self.mean = X.mean()

    def predict(self, X: pd.DataFrame):
        prd =  len(X) * [self.mean]
        return prd


class RollingMeanRegressor (RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win

    def fit(self, X: pd.DataFrame):
        None

    def predict(self, X: pd.DataFrame):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-self.win_size+1):i+1].mean()
        return prd