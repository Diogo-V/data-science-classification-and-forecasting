from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
import pandas as pd
from matplotlib.pyplot import subplots, savefig
from ds_charts import multiple_line_chart
from ts_functions import HEIGHT, PREDICTION_MEASURES, plot_evaluation_results
import matplotlib.pyplot as plt

class ARIMA:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def compute_arima(self) -> ARIMA_model:
        pred = ARIMA_model(self.data, order=(1, 1, 3))
        model = pred.fit(method_kwargs={'warn_convergence': False})
        model.plot_diagnostics(figsize=(2*HEIGHT, 2*HEIGHT))
        savefig(f'health-forecasting/records/evaluation/arima_diagnostics.png')
        return model

    def plot_forecasting_series(self, trn, tst, prd_trn, prd_tst, file_path: str, tittle: str, x_label: str = 'time', y_label:str =''):
      _, ax = plt.subplots(1,1,figsize=(5*HEIGHT, HEIGHT), squeeze=True)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.set_title(tittle)
      ax.plot(trn.index, trn, label='train', color='b')
      ax.plot(trn.index, prd_trn, '--y', label='train prediction')
      ax.plot(tst.index, tst, label='test', color='g')
      ax.plot(tst.index, prd_tst, '--r', label='test prediction')
      ax.legend(prop={'size': 5})
      savefig(file_path)

    def explore_arima(self, test: pd.DataFrame) -> None:
        measure = 'R2'
        flag_pct = False
        last_best = -100
        best = ('',  0, 0.0)
        best_model = None
        
        d_values = (0, 1, 2, 3, 4)
        params = (1, 2, 3, 5, 6, 7)
        ncols = len(d_values)
        
        fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)
        
        for der in range(len(d_values)):
            d = d_values[der]
            values = {}
            for q in params:
                yvalues = []
                for p in params:
                    pred = ARIMA_model(self.data, order=(p, d, q))
                    model = pred.fit(method_kwargs={'warn_convergence': False})
                    prd_tst = model.forecast(steps=len(test), signal_only=False)
                    yvalues.append(PREDICTION_MEASURES[measure](test,prd_tst))
                    if yvalues[-1] > last_best:
                        best = (p, d, q)
                        last_best = yvalues[-1]
                        best_model = model
                values[q] = yvalues
            multiple_line_chart(params, values, ax=axs[0, der], title=f'ARIMA d={d}', xlabel='p', ylabel=measure, percentage=flag_pct)

        savefig(f'health-forecasting/records/evaluation/arima_study.png')
        print(f'Best results achieved with (p,d,q)=({best[0]}, {best[1]}, {best[2]}) ==> measure={last_best:.2f}')

        best_model.plot_diagnostics(figsize=(2*HEIGHT, 2*HEIGHT))
        savefig(f'health-forecasting/records/evaluation/arima_diagnostics.png')

        prd_trn = best_model.predict(start=0, end=len(self.data)-1)
        prd_tst = best_model.forecast(steps=len(test))
        print(f'\t{measure}={PREDICTION_MEASURES[measure](test, prd_tst)}')
        
        plot_evaluation_results(self.data.values, prd_trn, test.values, prd_tst, f'health-forecasting/records/evaluation/arima_eval')
        self.plot_forecasting_series(self.data, test, prd_trn, prd_tst, f'health-forecasting/records/evaluation/arima_plots.png', "Arima plot", x_label='Date', y_label='Glucose')
    