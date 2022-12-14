from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT, plot_overfitting_study
from sklearn.metrics import accuracy_score, f1_score



HEIGHT: int = 4

class RTClassifier:
  
  def __init__(self, data_train: DataFrame, data_test: DataFrame) -> None:
    self.train_data = data_train
    self.train_y = self.train_data.pop('class').values
    self.test_data = data_test
    self.test_y = self.test_data.pop('class').values
    self.n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
    self.max_depths = [5, 10, 25]
    self.max_features = [.3, .5, .7, 1]
    

  def explore_best_rt(self):
    best = ('', 0, 0)
    last_best = 0
    best_model = None

    cols = len(self.max_depths)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(self.max_depths)):
        d = self.max_depths[k]
        values = {}
        for f in self.max_features:
            y_test_values = []
            y_train_values = []
            y_test_values_f1 = []
            y_train_values_f1 = []
            for n in self.n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(self.train_data, self.train_y)
                prd_tst_Y = rf.predict(self.test_data)
                prd_trn_Y = rf.predict(self.train_data)
                y_test_values.append(accuracy_score(self.test_y, prd_tst_Y))
                y_train_values.append(accuracy_score(self.train_y, prd_trn_Y))
                y_test_values_f1.append(f1_score(self.test_y, prd_tst_Y, average="macro"))
                y_train_values_f1.append(f1_score(self.train_y, prd_trn_Y, average="macro"))
                if y_test_values[-1] > last_best:
                    best = (d, f, n)
                    last_best = y_test_values[-1]
                    best_model = rf

                # plot_overfitting_study(self.n_estimators, y_train_values, y_test_values, name=f'RF_depth={d}_vars={f}', xlabel='nr_estimators', ylabel=str(accuracy_score))
                # plot_overfitting_study(self.n_estimators, y_train_values, y_test_values, name=f'RF_depth={d}_vars={f}', xlabel='nr_estimators', ylabel=str(f1_score))

            values[f] = y_test_values

    figure()
    multiple_line_chart(self.n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                        xlabel='nr estimators', ylabel='accuracy', percentage=True)
    savefig(f'climate/records/evaluation/rf_study.png')
    show()
    print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))