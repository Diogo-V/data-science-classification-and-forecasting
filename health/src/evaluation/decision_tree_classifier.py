from numpy import ndarray, argsort, arange
import numpy as np
import pandas as pd
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, subplots, savefig, show
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree as sklearn_tree
from sklearn.metrics import accuracy_score, f1_score, classification_report
from ds_charts import multiple_line_chart, plot_evaluation_results_2_train_test_matrixes, horizontal_bar_chart, plot_overfitting_study
from matplotlib.pyplot import imread, imshow, axis
import math

# criteria
# max depth
# minimum impurity decrease

HEIGHT: int = 4

class DTClassifier:
  
  def __init__(self, data_train: DataFrame, data_test: DataFrame) -> None:
    self.target = 'readmitted'
    self.data_train = data_train
    self.data_test = data_test
    self.trnY: ndarray = data_train.pop(self.target).values
    self.trnX: ndarray = data_train.values
    self.labels = unique(self.trnY)
    self.labels.sort()
    
    self.tstY: ndarray = data_test.pop(self.target).values
    self.tstX: ndarray = data_test.values

    self.min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    self.max_depths = [2, 5, 10, 15, 20, 25, 35, 40]
    self.criteria = ['entropy', 'gini']
    self.best = ('',  0, 0.0)
    self.last_best = 0
    self.best_model = None

  def compute_best_dt(self):
    fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(self.criteria)):
        f = self.criteria[k]
        values = {}
        for d in self.max_depths:
            yvalues = []
            for imp in self.min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(self.trnX, self.trnY)
                prdY = tree.predict(self.tstX)
                yvalues.append(accuracy_score(self.tstY, prdY))
                if yvalues[-1] > self.last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    self.best_model = tree

            values[d] = yvalues
        multiple_line_chart(self.min_impurity_decrease, values, ax=axs[0, k], title=f'Decision Trees with {f} criteria',
                               xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
    savefig(f'health/records/evaluation/dt_study.png')
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.6f ==> accuracy=%1.5f'%(best[0], best[1], best[2], last_best))

    return best[0], best[1], best[2]

  def explore_best_tree_graph_light(self):
    if self.best_model is None:
      print("You have to run the explore best dt before this function :)")
    labels = [str(value) for value in self.labels]
    sklearn_tree.plot_tree(self.best_model, feature_names=self.data_train.columns, class_names=labels)
    savefig('health/records/evaluation/dt_tree_light.png')

  def compute_dt_best_matrix_results(self, depth: int, criteria: str, impurity: int):
    print('Computing best Decision Tree results...')
    labels = pd.unique(self.tstY)
    labels.sort()

    labels_str=["Class 1", "Class 2", "Class 3"]	

    dt = DecisionTreeClassifier(max_depth=depth, criterion=criteria, min_impurity_decrease=impurity)
    dt.fit(self.trnX, self.trnY)
    prd_trn = dt.predict(self.trnX)
    prd_tst = dt.predict(self.data_test)
    train_acc = accuracy_score(self.trnY, prd_trn)
    test_acc = accuracy_score(self.tstY, prd_tst)
    error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

    plot_evaluation_results_2_train_test_matrixes(labels, self.trnY, prd_trn, self.tstY, prd_tst)
    savefig('health/records/evaluation/dt_depth_impurity_best_results.png')

    f= open('health/records/evaluation/dt_depth_impurity_best_results_details.txt', 'w')
    f.write(f'Best approach: Depth {depth} with criterion {criteria} and impurity decrease {impurity}\n')
    f.write("Accuracy Train: {:.5f}\n".format(train_acc))
    f.write("Accuracy Test: {:.5f}\n".format(test_acc))
    f.write("Diff between train and test: {:.5f}\n".format(train_acc - test_acc))
    f.write("Root mean squared error: {:.5f}\n".format(error))
    f.write("########################\n")
    f.write("Train\n")
    f.write(classification_report(self.trnY, prd_trn,target_names=labels_str))
    f.write("Test\n")
    f.write(classification_report(self.tstY, prd_tst,target_names=labels_str))

  def compute_dt_feature_importance(self):
    if self.best_model is None:
      print("You have to run the explore best dt before this function :)")
    
    print("Computing feature importance in decision tree...")
    variables = self.data_train.columns
    importances = self.best_model.feature_importances_
    indices = argsort(importances)[::-1]
    elems = []
    imp_values = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f'{f+1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    horizontal_bar_chart(elems, imp_values, error=None, title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
    savefig(f'health/records/evaluation/dt_feature_ranking.png')

  def compute_best_dt_overfit(self, criteria: str, impurity: int):

    def plot_overfitting_study(t, xvalues, prd_trn, prd_tst, name, xlabel, ylabel, pct=True):
      evals = {'Train': prd_trn, 'Test': prd_tst}
      figure()
      multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=pct)
      savefig(f'health/records/evaluation/dt_overfitting_exploration_{t}.png')

    print("Computing best decision tree overfitting....")
    imp = impurity
    f = criteria
    y_tst_values = []
    y_trn_values = []
    y_test_values_f1 = []
    y_train_values_f1 = []
    figure()
    for d in self.max_depths:
        tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
        tree.fit(self.trnX, self.trnY)
        prdY = tree.predict(self.tstX)
        prd_tst_Y = tree.predict(self.tstX)
        prd_trn_Y = tree.predict(self.trnX)
        y_tst_values.append(accuracy_score(self.tstY, prd_tst_Y))
        y_trn_values.append(accuracy_score(self.trnY, prd_trn_Y))
        y_test_values_f1.append(f1_score(self.tstY, prd_tst_Y, average="macro"))
        y_train_values_f1.append(f1_score(self.trnY, prd_trn_Y, average="macro"))
    plot_overfitting_study('accuracy', self.max_depths, y_trn_values, y_tst_values, name=f'DT=imp{imp}_{f}_accuracy', xlabel='max_depth', ylabel=str(accuracy_score))
    plot_overfitting_study('f1', self.max_depths, y_train_values_f1, y_test_values_f1, name=f'DT=imp{imp}_{f}_f1', xlabel='max_depth', ylabel=str(f1_score))