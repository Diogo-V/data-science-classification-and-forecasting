from numpy import ndarray, std, argsort
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.ensemble import RandomForestClassifier
from ds_charts import plot_evaluation_results_2_train_test_matrixes, multiple_line_chart, horizontal_bar_chart, HEIGHT, plot_overfitting_study
from sklearn.metrics import accuracy_score, f1_score



HEIGHT: int = 4

class RTClassifier:
  
  def __init__(self, data_train: DataFrame, data_test: DataFrame) -> None:
    self.train_data = data_train
    self.train_y = self.train_data.pop('readmitted').values
    self.test_data = data_test
    self.test_y = self.test_data.pop('readmitted').values
    self.n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
    self.max_depths = [5, 10, 25]
    self.max_features = [.3, .5, .7, 1]
    

  def explore_best_rt(self):
    best = ('', 0, 0)
    last_best = 0

    cols = len(self.max_depths)
    figure()
    fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
    for k in range(len(self.max_depths)):
        d = self.max_depths[k]
        print(f"Depth: {d}")
        values = {}
        for f in self.max_features:
            print(f"Features: {f}")
            y_test_values = []
            y_train_values = []
            y_test_values_f1 = []
            y_train_values_f1 = []
            for n in self.n_estimators:
                print(f"Estimators: {n}")
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

            values[f] = y_test_values

        multiple_line_chart(self.n_estimators, values, ax=axs[0, k], title=f'Random Forests with max_depth={d}',
                        xlabel='nr estimators', ylabel='accuracy', percentage=True)
    savefig(f'health/records/evaluation/rf_study.png')
    print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
    f= open('health/records/evaluation/rf_best_details.txt', 'w')
    f.write(f'Best approach: Max Depth = {best[0]} with {best[1]} features and {best[2]} estimators\n')
    f.close()

    return best[0], best[1], best[2]

  def compute_best_rt_results(self, depth, features, estimators):
    
    rf = RandomForestClassifier(n_estimators=estimators, max_depth=depth, max_features=features)
    rf.fit(self.train_data, self.train_y)
    prd_trn = rf.predict(self.train_data)
    prd_tst = rf.predict(self.test_data)

    # First, plot general results
    labels = unique(self.train_y)
    labels.sort()
    figure()
    plot_evaluation_results_2_train_test_matrixes(labels, self.train_y, prd_trn, self.test_y, prd_tst)
    savefig(f'health/records/evaluation/rf_best_results.png')

    # Then, plot features importance
    variables = self.train_data.columns
    print(variables)
    importances = rf.feature_importances_
    stdevs = std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = argsort(importances)[::-1]
    elems = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')
    figure()
    horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Random Forest Features importance', xlabel='importance', ylabel='variables')
    savefig(f'health/records/evaluation/rf_best_feature_rankings.png')

    # Then, plot overfitting
    y_tst_values = []
    y_trn_values = []
    y_tst_values_f1 = []
    y_trn_values_f1 = []
  
    for n in self.n_estimators:
      print(f"Overfitting for {n} estimators")
      rf = RandomForestClassifier(n_estimators=n, max_depth=depth, max_features=features)
      rf.fit(self.train_data, self.train_y)
      prd_tst_Y = rf.predict(self.test_data)
      prd_trn_Y = rf.predict(self.train_data)
      y_tst_values.append(accuracy_score(self.test_y, prd_tst_Y))
      y_trn_values.append(accuracy_score(self.train_y, prd_trn_Y))
      y_tst_values_f1.append(f1_score(self.test_y, prd_tst_Y, average="macro"))
      y_trn_values_f1.append(f1_score(self.train_y, prd_trn_Y, average="macro"))
    self.plot_overfitting_study(self.n_estimators, y_trn_values, y_tst_values, name=f'RF_depth={depth}_vars={features}', xlabel='nr_estimators', ylabel="accuracy score")
    self.plot_overfitting_study(self.n_estimators, y_trn_values_f1, y_tst_values_f1, name=f'RF_depth={depth}_vars={features}_f1', xlabel='nr_estimators', ylabel="f1 score")

  def plot_overfitting_study(self, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
    evals = {'Train': prd_trn, 'Test': prd_tst}
    figure()
    multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
    savefig(f'health/records/evaluation/{name}_overfitting_exploration.png')