import pandas as pd
from matplotlib.pyplot import figure, xticks, savefig
from ts_functions import plot_series, HEIGHT
import matplotlib.pyplot as plt
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, HEIGHT
from sklearn.base import RegressorMixin

class Aggregation:
    def __init__(self, data: pd.DataFrame) -> None:
      self.data: pd.DataFrame = data

    def compute_aggregation(self) -> pd.DataFrame:
        return self.compute_aggregate_daily(self.data)  # Seems like it's the best one

    def explore_aggregation(self) -> pd.DataFrame:
        daily = self.compute_aggregate_daily(self.data)
        weekly = self.compute_aggregate_weekly(self.data)
        monthly = self.compute_aggregate_monthly(self.data)

        # Plots figures for each aggregation measure
        self.plot_figure(daily, "daily")
        self.plot_figure(weekly, "weekly")
        self.plot_figure(monthly, "monthly")

        # Evaluate with Persistance
        print("EVALUATING WITH PERSISTANCE...")
        self.persistance(daily, "daily")
        self.persistance(weekly, "weekly")
        self.persistance(monthly, "monthly")


    def aggregate_by(self, data: pd.Series, index_var: str, period: str):
        index = data.index.to_period(period)
        agg_df = data.copy().groupby(index).mean()
        agg_df[index_var] = index.drop_duplicates().to_timestamp()
        agg_df.set_index(index_var, drop=True, inplace=True)
        return agg_df

    def plot_figure(self, data: pd.DataFrame, title: str):
        figure(figsize=(3*HEIGHT, HEIGHT))
        plot_series(data, title=title, x_label='Date', y_label='Glucose')
        xticks(rotation = 45)
        savefig(f'health-forecasting/records/transformation/aggregation/aggregation_{title}_plot.png')

    def compute_aggregate_daily(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate_by(data, "Date", "D")

    def compute_aggregate_weekly(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate_by(data, "Date", "W")

    def compute_aggregate_monthly(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate_by(data, "Date", "M")

    def split_dataframe(self, data: pd.DataFrame, trn_pct=0.70):
        trn_size = int(len(data) * trn_pct)
        df_cp = data.copy()
        train: pd.DataFrame = df_cp.iloc[:trn_size, :]
        test: pd.DataFrame = df_cp.iloc[trn_size:]
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

    def persistance(self, data: pd.DataFrame, approach: str) -> None:
    
        train, test = self.split_dataframe(data, trn_pct=0.75)
        eval_results = {}

        fr_mod = PersistenceRegressor()
        fr_mod.fit(train)
        prd_trn = fr_mod.predict(train)
        prd_tst = fr_mod.predict(test)

        measure = "R2"
        eval_results['Persistance'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
        print(eval_results)
        
        plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'health-forecasting/records/transformation/aggregation/aggregation_{approach}_{measure}_persistance_eval.png')
        plt.savefig(f'health-forecasting/records/transformation/aggregation/aggregation_{approach}_{measure}_persistance_eval.png')
        self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'health-forecasting/records/transformation/aggregation/aggregation_{approach}_{measure}_persistance_plots.png', x_label="Date", y_label="Glucose")
        plt.savefig(f'health-forecasting/records/transformation/aggregation/aggregation_{approach}_{measure}_persistance_plots.png')

class PersistenceRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: pd.DataFrame):
        self.last = X.iloc[-1,0]
        print(self.last)

    def predict(self, X: pd.DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd