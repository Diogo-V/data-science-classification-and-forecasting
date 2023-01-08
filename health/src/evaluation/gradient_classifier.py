from numpy import ndarray, argsort, std
import pandas as pd
import math
import numpy as np
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from ds_charts import plot_evaluation_results_2_train_test_matrixes, plot_evaluation_results, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score, f1_score

class GradientClassifier:
    
    def __init__(self, data_train: pd.DataFrame, data_test: pd.DataFrame) -> None:
        self.train_data = data_train
        self.train_y = self.train_data.pop('readmitted').values
        self.test_data = data_test
        self.test_y = self.test_data.pop('readmitted').values
        self.n_estimators = [5, 10, 25, 50, 75, 100, 200, 300, 400]
        self.max_depths = [5, 10, 25]
        self.learning_rate = [.1, .5, .9]

    def explore_best_gradient(self):
        best = ('', 0, 0)
        last_best = 0

        cols = len(self.max_depths)
        figure()
        fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
        for k in range(len(self.max_depths)):
            d = self.max_depths[k]
            values = {}
            for lr in self.learning_rate:
                yvalues = []
                for n in self.n_estimators:
                    print(f'Classifying with {d} max depth, {lr} learning rate, {n} estimators')
                    gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                    gb.fit(self.train_data, self.train_y)
                    prdY = gb.predict(self.test_data)
                    yvalues.append(accuracy_score(self.test_y, prdY))
                    if yvalues[-1] > last_best:
                        best = (d, lr, n)
                        last_best = yvalues[-1]
                        best_model = gb
                values[lr] = yvalues
            multiple_line_chart(self.n_estimators, values, ax=axs[0, k], title=f'Gradient Boorsting with max_depth={d}',
                                xlabel='nr estimators', ylabel='Accuracy', percentage=True)
        savefig(f'health/records/evaluation/gradient_boosting_study.png')
        print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
        f= open('health/records/evaluation/gb_best_details.txt', 'w')
        f.write(f'Best approach: Max Depth = {best[0]} with learning rate = {best[1]} and estimators = {best[2]}')
        f.close()

        return best[0], best[1], best[2]

    def compute_best_gradient(self, max_depth, lr, estimators):
        gb = GradientBoostingClassifier(n_estimators=estimators, max_depth=max_depth, learning_rate=lr)
        gb.fit(self.train_data, self.train_y)
        prd_trn = gb.predict(self.train_data)
        prd_tst = gb.predict(self.test_data)

        # First, plot general results
        labels = unique(self.train_y)
        labels.sort()
        figure()
        plot_evaluation_results_2_train_test_matrixes(labels, self.train_y, prd_trn, self.test_y, prd_tst)
        savefig(f'health/records/evaluation/gb_best_results.png')

        # Then, plot features importance
        variables = self.train_data.columns
        importances = gb.feature_importances_
        indices = argsort(importances)[::-1]
        stdevs = std([tree[0].feature_importances_ for tree in gb.estimators_], axis=0)
        elems = []
        for f in range(len(variables)):
            elems += [variables[indices[f]]]
        figure()
        horizontal_bar_chart(elems, importances[indices], stdevs[indices], title='Gradient Boosting Features importance', xlabel='importance', ylabel='variables')
        savefig(f'health/records/evaluation/gb_best_feature_rankings.png')

         # Then, plot overfitting
        y_tst_values = []
        y_trn_values = []
        y_tst_values_f1 = []
        y_trn_values_f1 = []
    
        for n in self.n_estimators:
            print(f"Overfitting for {n} estimators")
            gb = GradientBoostingClassifier(n_estimators=n, max_depth=max_depth, learning_rate=lr)
            gb.fit(self.train_data, self.train_y)
            prd_tst_Y = gb.predict(self.test_data)
            prd_trn_Y = gb.predict(self.train_data)
            y_tst_values.append(accuracy_score(self.test_y, prd_tst_Y))
            y_trn_values.append(accuracy_score(self.train_y, prd_trn_Y))
            y_tst_values_f1.append(f1_score(self.test_y, prd_tst_Y, average="macro"))
            y_trn_values_f1.append(f1_score(self.train_y, prd_trn_Y, average="macro"))
        self.plot_overfitting_study(self.n_estimators, y_trn_values, y_tst_values, name=f'GB_depth={max_depth}_lr={lr}_estimators={estimators}', xlabel='nr_estimators', ylabel="accuracy score")
        self.plot_overfitting_study(self.n_estimators, y_trn_values_f1, y_tst_values_f1, name=f'GB_depth={max_depth}_lr={lr}_estimators={estimators}_f1', xlabel='nr_estimators', ylabel="f1 score")

    def plot_overfitting_study(self, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
        evals = {'Train': prd_trn, 'Test': prd_tst}
        figure()
        multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
        savefig(f'health/records/evaluation/{name}_overfitting_exploration.png')
