import pandas as pd
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, HEIGHT
from matplotlib.pyplot import subplots, savefig

class PersistenceRegressor (RegressorMixin):
    def __init__(self, train: pd.DataFrame, test:pd.DataFrame):
        super().__init__()
        self.mean = 0
        self.train_data = train
        self.test_data = test

        self.last = 0


    def explore_rolling_mean_regressor(self):
      measure = 'R2'
      eval_results = {}

      self.fit(self.train_data)

      prd_trn = self.predict(self.train_data)
      prd_tst = self.predict(self.test_data)

      eval_results['Persistence'] = PREDICTION_MEASURES[measure](self.test_data.values, prd_tst)
      f = open(f'health-forecasting/records/evaluation/health_persistence_results.txt', 'w')
      f.write(f'{eval_results}')

      plot_evaluation_results(self.train_data.values, prd_trn, self.test_data.values, prd_tst, 'Persistence', f'health-forecasting/records/evaluation/health_persistence_eval')
      self.plot_forecasting_series(self.train_data, self.test_data, prd_trn, prd_tst, 'Persistence', f'health-forecasting/records/evaluation/health_persistence_plots.png', x_label='timestamp', y_label='Glucose')


    def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, figtittle: str, figname: str, x_label: str = 'time', y_label:str =''):
      _, ax = subplots(1,1,figsize=(6*HEIGHT, HEIGHT), squeeze=True)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.set_title(figtittle)
      ax.plot(trn.index, trn, label='train', color='b')
      ax.plot(trn.index, prd_trn, '--y', label='train prediction')
      ax.plot(tst.index, tst, label='test', color='g')
      ax.plot(tst.index, prd_tst, '--r', label='test prediction')
      ax.legend(prop={'size': 5})

      savefig(figname)


    def fit(self, X: pd.DataFrame):
        self.last = X.iloc[-1,0]
        print(self.last)

    def predict(self, X: pd.DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd


