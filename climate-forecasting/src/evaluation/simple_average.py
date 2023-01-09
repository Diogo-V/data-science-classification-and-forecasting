import pandas as pd
from sklearn.base import RegressorMixin
from ts_functions import PREDICTION_MEASURES, plot_evaluation_results, HEIGHT
from matplotlib.pyplot import figure, subplots, savefig

class SimpleAvgRegressor (RegressorMixin):
    def __init__(self, train: pd.DataFrame, test:pd.DataFrame):
        super().__init__()
        self.mean = 0
        self.train_data = train
        self.test_data = test


    def compute_simple_avg_regressor(self):
      measure = 'R2'
      eval_results = {}

      self.fit(self.train_data)
      prd_trn = self.predict(self.train_data)
      prd_tst = self.predict(self.test_data)

      eval_results['SimpleAvg'] = PREDICTION_MEASURES[measure](self.test_data.values, prd_tst)
      f = open(f'climate-forecasting/records/evaluation/simple_average_results.txt', 'w')
      f.write(f'{eval_results}')

      plot_evaluation_results(self.train_data.values, prd_trn, self.test_data.values, prd_tst, f'climate-forecasting/records/evaluation/climate_simpleAvg_eval.png')
      self.plot_forecasting_series(self.train_data, self.test_data, prd_trn, prd_tst, 'climate-forecasting/records/evaluation/climate_simpleAvg_plots.png', x_label='timestamp', y_label='QV2M')


    def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, figname: str, x_label: str = 'time', y_label:str =''):
      _, ax = subplots(1,1,figsize=(6*HEIGHT, HEIGHT), squeeze=True)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.set_title(figname)
      ax.plot(trn.index, trn, label='train', color='b')
      ax.plot(trn.index, prd_trn, '--y', label='train prediction')
      ax.plot(tst.index, tst, label='test', color='g')
      ax.plot(tst.index, prd_tst, '--r', label='test prediction')
      ax.legend(prop={'size': 5})

      savefig(figname)


    def fit(self, X: pd.DataFrame):
        self.mean = X.mean()

    def predict(self, X: pd.DataFrame):
        prd =  len(X) * [self.mean]
        return prd


