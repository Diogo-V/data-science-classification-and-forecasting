import pandas as pd
from matplotlib.pyplot import figure, xticks, savefig, subplots
from ts_functions import plot_series, HEIGHT, split_dataframe, PREDICTION_MEASURES, plot_evaluation_results
from sklearn.base import RegressorMixin


class Smoothing:

  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data
    
    self.win_sizes = [7, 14, 21, 30, 60, 90]

    self.best_size = 90

  
  def compute_smoothing(self) -> pd.DataFrame:
    rolling = self.data.rolling(window=self.best_size)
    smooth_df = rolling.mean()
    smooth_df = smooth_df.dropna()
    return smooth_df

  def explore_smoothing(self) -> None:
    for size in self.win_sizes:
        rolling = self.data.rolling(window=size)
        smooth_df = rolling.mean()
        smooth_df = smooth_df.dropna()
        figure(figsize=(3*HEIGHT, HEIGHT))
        plot_series(smooth_df, title=f'Smoothing (win_size={size})', x_label='timestamp', y_label='QV2M')
        xticks(rotation = 45)
        savefig(f'climate-forecasting/records/transformation/smoothing/smoothing_explore_{size}.png')
        self.persistence_regressor(smooth_df, size)


  def split_dataframe(data, trn_pct=0.70):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train: pd.DataFrame = df_cp.iloc[:trn_size, :]
    test: pd.DataFrame = df_cp.iloc[trn_size:]
    return train, test

  def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, file_path: str, tittle: str, x_label: str = 'time', y_label:str =''):
    _, ax = subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(tittle)
    ax.plot(trn.index, trn, label='train', color='b')
    ax.plot(trn.index, prd_trn, '--y', label='train prediction')
    ax.plot(tst.index, tst, label='test', color='g')
    ax.plot(tst.index, prd_tst, '--r', label='test prediction')
    ax.legend(prop={'size': 5})

    savefig(file_path)


  def persistence_regressor(self, data: pd.DataFrame, size: str):
    train, test = split_dataframe(data, trn_pct=0.75)

    eval_results = {}
    measure = 'R2'

    fr_mod = PersistenceRegressor()
    fr_mod.fit(train)
    prd_trn = fr_mod.predict(train)
    prd_tst = fr_mod.predict(test)

    eval_results['Persistence'] = PREDICTION_MEASURES[measure](test.values, prd_tst)
 
    f = open(f'climate-forecasting/records/transformation/smoothing/smoothing_persistence_eval_{size}.txt', 'w')
    f.write(f"{size} {eval_results}")

    plot_evaluation_results(train.values, prd_trn, test.values, prd_tst, f'climate-forecasting/records/transformation/smoothing/smoothing_persistence_eval_{size}')
    self.plot_forecasting_series(train, test, prd_trn, prd_tst, f'climate-forecasting/records/transformation/smoothing/smoothing_persistence_plots_{size}.png', f'Persistence plots with {size}', y_label='QV2M')



class PersistenceRegressor (RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0

    def fit(self, X: pd.DataFrame):
        self.last = X.iloc[-1,0]

    def predict(self, X: pd.DataFrame):
        prd = X.shift().values
        prd[0] = self.last
        return prd

