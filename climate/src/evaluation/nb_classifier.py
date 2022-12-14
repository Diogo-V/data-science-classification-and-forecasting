import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from numpy import ndarray
from pandas import unique
from matplotlib.pyplot import figure, savefig, show, subplots
from ds_charts import plot_evaluation_results_train_test_matrixes, bar_chart
import math
import numpy as np

HEIGHT: int = 4

class NBClassifier:
  
  def __init__(self, data_train: pd.DataFrame, data_test: pd.DataFrame) -> None:
    self.data_train = data_train
    self.data_test = data_test
    self.target = "class"

    self.estimators = {'GaussianNB': GaussianNB(),
                      'MultinomialNB': MultinomialNB(),
                      'BernoulliNB': BernoulliNB(),
                      #'CategoricalNB': CategoricalNB(),
                      # 'ComplementNB': ComplementNB()
                      }

  def compute_nb_best_results(self):
    trnY: ndarray = self.data_train.pop(self.target).values
    trnX: ndarray = self.data_train.values

    labels = unique(trnY)
    labels.sort()

    labels_str=["Class 1", "Class 2"]	

    tstY: ndarray = self.data_test.pop(self.target).values
    tstX: ndarray = self.data_test.values

    clf = BernoulliNB()
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    train_acc = accuracy_score(trnY, prd_trn)
    test_acc = accuracy_score(tstY, prd_tst)
    error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)


    plot_evaluation_results_train_test_matrixes(labels, trnY, prd_trn, tstY, prd_tst)
    savefig('climate/records/evaluation/nb_best.png')

    f= open('climate/records/evaluation/nb_best_details.txt', 'w')
    f.write("Accuracy Train: {:.5f}\n".format(train_acc))
    f.write("Accuracy Test: {:.5f}\n".format(test_acc))
    f.write("Diff between train and test: {:.5f}\n".format(train_acc - test_acc))
    f.write("Root mean squared error: {:.5f}\n".format(error))
    f.write("########################\n")
    f.write("Train\n")
    f.write(classification_report(trnY, prd_trn,target_names=labels_str))
    f.write("Test\n")
    f.write(classification_report(tstY, prd_tst,target_names=labels_str))


  def explore_best_nb_value(self):

    trnY: ndarray = self.data_train.pop(self.target).values
    trnX: ndarray = self.data_train.values

    labels = unique(trnY)
    labels.sort()

    tstY: ndarray = self.data_test.pop(self.target).values
    tstX: ndarray = self.data_test.values

    xvalues = []
    yvalues_acc_score = []
    yvalues_f1_score = []

    for clf in self.estimators:
        xvalues.append(clf)
        self.estimators[clf].fit(trnX, trnY)
        prdY = self.estimators[clf].predict(tstX)
        yvalues_acc_score.append(accuracy_score(tstY, prdY))
        yvalues_f1_score.append(f1_score(tstY, prdY))

    _, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    bar_chart(xvalues, yvalues_acc_score, ax=axs[0], title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
    bar_chart(xvalues, yvalues_f1_score, ax=axs[1], title='Comparison of Naive Bayes Models', ylabel='f1_score', percentage=True)
    savefig('climate/records/evaluation/nb_study.png')


