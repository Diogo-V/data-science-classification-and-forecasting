import pandas as pd
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, HEIGHT
from matplotlib.pyplot import subplots, savefig

class RollingMeanRegressor (RegressorMixin):
    def __init__(self, train: pd.DataFrame, test:pd.DataFrame):
        super().__init__()
        self.mean = 0
        self.train_data = train
        self.test_data = test

        self.win_sizes = [2, 3, 4, 5, 7, 8, 10, 15, 20]

        self.best_size = 2

    def compute_rolling_mean_regressor(self):
      measure = 'R2'
      eval_results = {}

      prd_trn = self.predict(self.train_data, self.best_size)
      prd_tst = self.predict(self.test_data, self.best_size)

      eval_results['RollingMean'] = PREDICTION_MEASURES[measure](self.test_data.values, prd_tst)
      f = open(f'health-forecasting/records/evaluation/health_rolling_mean_best_{self.best_size}_results.txt', 'w')
      f.write(f'{self.best_size} {eval_results}')

      plot_evaluation_results(self.train_data.values, prd_trn, self.test_data.values, prd_tst, f'health-forecasting/records/evaluation/health_rolling_mean_{self.best_size}_eval')
      self.plot_forecasting_series(self.train_data, self.test_data, prd_trn, prd_tst,  f'{self.best_size}', f'health-forecasting/records/evaluation/health_rolling_mean_{self.best_size}_plots.png', x_label='timestamp', y_label='Glucose')


    def explore_rolling_mean_regressor(self):
      measure = 'R2'
      eval_results = {}

      self.fit(self.train_data)

      for size in self.win_sizes:
        prd_trn = self.predict(self.train_data, size)
        prd_tst = self.predict(self.test_data, size)

        eval_results['RollingMean'] = PREDICTION_MEASURES[measure](self.test_data.values, prd_tst)
        f = open(f'health-forecasting/records/evaluation/health_rolling_mean_{size}_results.txt', 'w')
        f.write(f'{size} {eval_results}')

        plot_evaluation_results(self.train_data.values, prd_trn, self.test_data.values, prd_tst, size, f'health-forecasting/records/evaluation/health_rolling_mean_{size}_eval', )
        self.plot_forecasting_series(self.train_data, self.test_data, prd_trn, prd_tst, f'{size}', f'health-forecasting/records/evaluation/health_rolling_mean_{size}_plots.png', x_label='timestamp', y_label='Glucose')


    def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, figname: str, figpath: str, x_label: str = 'time', y_label:str =''):
      _, ax = subplots(1,1,figsize=(6*HEIGHT, HEIGHT), squeeze=True)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.set_title(figname)
      ax.plot(trn.index, trn, label='train', color='b')
      ax.plot(trn.index, prd_trn, '--y', label='train prediction')
      ax.plot(tst.index, tst, label='test', color='g')
      ax.plot(tst.index, prd_tst, '--r', label='test prediction')
      ax.legend(prop={'size': 5})

      savefig(figpath)


    def fit(self, X: pd.DataFrame):
        None

    def predict(self, X: pd.DataFrame, size: int):
        prd = len(X) * [0]
        for i in range(len(X)):
            prd[i] = X[max(0, i-size+1):i+1].mean()
        return prd


