import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.metrics import accuracy_score
from numpy import ndarray
from pandas import DataFrame, read_csv, unique
from matplotlib.pyplot import figure, savefig, show
from ds_charts import plot_evaluation_results, bar_chart

estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()
              #'CategoricalNB': CategoricalNB
              }

target = 'class'

class NBClassifier:
  
  def __init__(self, data_train: pd.DataFrame, data_test: pd.DataFrame) -> None:
    self.data_train = data_train
    self.data_test = data_test

  def evaluate_nb(self):
    trnY: ndarray = self.data_train.pop(target).values
    trnX: ndarray = self.data_train.values

    labels = unique(trnY)
    labels.sort()

    tstY: ndarray = self.data_test.pop(target).values
    tstX: ndarray = self.data_test.values

    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    savefig('images/{file_tag}_nb_best.png')
    show()

  def explore_nb(self):
    trnY: ndarray = self.data_train.pop(target).values
    trnX: ndarray = self.data_train.values

    labels = unique(trnY)
    labels.sort()

    tstY: ndarray = self.data_test.pop(target).values
    tstX: ndarray = self.data_test.values

    xvalues = []
    yvalues = []

    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(accuracy_score(tstY, prdY))

    figure()
    bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
    savefig(f'health/records/evaluation/nb_study.png')
    show()