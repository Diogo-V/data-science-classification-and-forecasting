from numpy import ndarray
import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show, plot, xlabel, ylabel
from sklearn.neural_network import MLPClassifier
from ds_charts import plot_evaluation_results, plot_evaluation_results_2_train_test_matrixes, multiple_line_chart, horizontal_bar_chart, HEIGHT
from sklearn.metrics import accuracy_score, f1_score
import math

target = 'readmitted'

class MLPClassify:

    def __init__(self, data_train: pd.DataFrame, data_test: pd.DataFrame) -> None:
        self.data_train = data_train
        self.data_test = data_test

    def explore_best_mlp(self):
        trnY: ndarray = self.data_train.pop(target).values
        trnX: ndarray = self.data_train.values

        labels = unique(trnY)
        labels.sort()

        labels_str=["Class 1", "Class 2", "Class 3"]	

        tstY: ndarray = self.data_test.pop(target).values
        tstX: ndarray = self.data_test.values

        lr_type = ['constant', 'invscaling', 'adaptive']
        max_iter = [100, 300, 500, 750, 1000, 2500, 5000]
        learning_rate = [.1, .5, .9]
        best = ('', 0, 0)
        last_best = 0
        best_model = None

        cols = len(lr_type)
        figure()
        fig, axs = subplots(1, cols, figsize=(cols*HEIGHT, HEIGHT), squeeze=False)
        for k in range(len(lr_type)):
            d = lr_type[k]
            values = {}
            for lr in learning_rate:
                yvalues = []
                for n in max_iter:
                    print(f'Classifying with {d} lr_type, {lr} learning rate, {n} max iterations')
                    mlp = MLPClassifier(activation='logistic', solver='sgd', learning_rate=d,
                                        learning_rate_init=lr, max_iter=n, verbose=False)
                    mlp.fit(trnX, trnY)
                    prdY = mlp.predict(tstX)
                    yvalues.append(accuracy_score(tstY, prdY))
                    if yvalues[-1] > last_best:
                        best = (d, lr, n)
                        last_best = yvalues[-1]
                        best_model = mlp
                values[lr] = yvalues
            multiple_line_chart(max_iter, values, ax=axs[0, k], title=f'MLP with lr_type={d}',
                                xlabel='mx iter', ylabel='Accuracy', percentage=True)
        savefig('health/records/evaluation/mlp_study.png')
        f= open('health/records/evaluation/mlp_best_details.txt', 'w')
        f.write(f'Best results with lr_type={best[0]}, learning rate={best[1]} and {best[2]} max iter, with accuracy={last_best}')
        f.close()

        return best[0], best[1], best[2]

    def compute_mlp_best_results(self, lr_type, learning_rate, max_iter):
        trnY: ndarray = self.data_train.pop(target).values
        trnX: ndarray = self.data_train.values

        tstY: ndarray = self.data_test.pop(target).values
        tstX: ndarray = self.data_test.values

        clf = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type,
                                        learning_rate_init=learning_rate, max_iter=max_iter, verbose=False)

        clf.fit(trnX, trnY)
        prd_trn = clf.predict(trnX)
        prd_tst = clf.predict(tstX)
        train_acc = accuracy_score(trnY, prd_trn)
        test_acc = accuracy_score(tstY, prd_tst)
        error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

        # First, plot general results
        labels = unique(trnY)
        labels.sort()
        figure()
        plot_evaluation_results_2_train_test_matrixes(labels, trnY, prd_trn, tstY, prd_tst)
        savefig('health/records/evaluation/mlp_best_metrics.png')

        # Then, plot loss curve
        figure()
        plot(clf.loss_curve_)
        xlabel("Iteration")
        ylabel("Loss")
        savefig('health/records/evaluation/mlp_loss_curve.png')

        # Then, plot overfitting
        y_tst_values = []
        y_trn_values = []
        y_tst_values_f1 = []
        y_trn_values_f1 = []

        max_iter = [100, 300, 500, 750, 1000, 2500, 5000]

        for n in max_iter:
            print(f"Overfitting for {n} epochs")
            clf = MLPClassifier(activation='logistic', solver='sgd', learning_rate=lr_type,
                                        learning_rate_init=learning_rate, max_iter=n, verbose=False)
            clf.fit(trnX, trnY)
            prd_tst_Y = clf.predict(tstX)
            prd_trn_Y = clf.predict(trnX)
            y_tst_values.append(accuracy_score(tstY, prd_tst_Y))
            y_trn_values.append(accuracy_score(trnY, prd_trn_Y))
            y_tst_values_f1.append(f1_score(tstY, prd_tst_Y, average="macro"))
            y_trn_values_f1.append(f1_score(trnY, prd_trn_Y, average="macro"))
        self.plot_overfitting_study(max_iter, y_trn_values, y_tst_values, name=f'MLP_lr_type={lr_type}_lr={learning_rate}', xlabel='nr_epochs', ylabel="Accuracy Score")
        self.plot_overfitting_study(max_iter, y_trn_values_f1, y_tst_values_f1, name=f'MLP_lr_type={lr_type}_lr={learning_rate}_f1', xlabel='nr_estimators', ylabel="F1 Score")

    def plot_overfitting_study(self, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
        evals = {'Train': prd_trn, 'Test': prd_tst}
        figure()
        multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
        savefig(f'health/records/evaluation/{name}_overfitting_exploration.png')

        



