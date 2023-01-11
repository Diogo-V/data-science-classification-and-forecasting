import pandas as pd
from matplotlib.pyplot import figure, show
from ds_charts import bar_chart
from imblearn.over_sampling import SMOTENC
from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np
from ds_charts import plot_evaluation_results
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Balancing:
  
  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data
    self.DETERMINISTIC_FACTOR = 3

  def compute_balancing(self) -> pd.DataFrame:
    self.data = self.explore_smote(self.data)
    return self.data

  def get_symbolic_index_array(self):
    headers = ['fips', 'SQ1', 'SQ3', 'SQ4']
    result = []
    for h in headers:
      result.append(self.data.columns.get_loc(h))
    return result

  def explore_balancing(self, data: pd.DataFrame) -> pd.DataFrame:
    
    data1 = data.copy(deep=True)

    smote = self.explore_smote(data)
    under = self.explore_under(data1)

    smote_acc = self.evaluate_knn(smote, 'smote')
    print(smote_acc)

    under_acc = self.evaluate_knn(under, 'under')
    print(under_acc)

    return smote if smote_acc > under_acc else under
  
  def evaluate_knn(self, dataset: pd.DataFrame, approach: str):
    data = dataset.copy(deep=True)
    y = data.pop('class').values
    X = data.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=self.DETERMINISTIC_FACTOR)

    labels = pd.unique(y)
    labels.sort()

    labels_str=["1", "2"]	

    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train, y_train)
    prd_train = knn.predict(X_train)
    prd_tst = knn.predict(X_test)
    train_acc = accuracy_score(y_train, prd_train)
    test_acc = accuracy_score(  y_test, prd_tst)
    error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

    plot_evaluation_results(labels, y_train, prd_train, y_test, prd_tst)
    plt.savefig(f'climate/records/preparation/balancing_{approach}_knn_results.png')

    f= open(f'climate/records/preparation/balancing_{approach}_knn_results_details.txt', 'w')
    f.write("Accuracy Train: {:.5f}\n".format(train_acc))
    f.write("Accuracy Test: {:.5f}\n".format(test_acc))
    f.write("Diff between train and test: {:.5f}\n".format(train_acc - test_acc))
    f.write("Root mean squared error: {:.5f}\n".format(error))
    f.write("########################\n")
    f.write("Train\n")
    f.write(classification_report(y_train, prd_train,target_names=labels_str))
    f.write("Test\n")
    f.write(classification_report(y_test, prd_tst,target_names=labels_str))
    
    return test_acc

  def explore_under(self, data: pd.DataFrame) -> pd.DataFrame:
    class_var = 'class'
    target_count = data[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    df_positives = data[data[class_var] == positive_class]
    df_negatives = data[data[class_var] == negative_class]

    df_neg_sample = pd.DataFrame(df_negatives.sample(len(df_positives)))
    df_under = pd.concat([df_positives, df_neg_sample], axis=0)
    values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
    print('Minority class=', positive_class, ':', len(df_positives))
    print('Majority class=', negative_class, ':', len(df_neg_sample))
    print('Proportion:', round(len(df_positives) / len(df_neg_sample), 2), ': 1')

    print(df_under[class_var].value_counts())

    return df_under

  def explore_smote(self, data: pd.DataFrame) -> pd.DataFrame:

    class_var = 'class'
    target_count = data[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    smote = SMOTENC(self.get_symbolic_index_array(), sampling_strategy='auto', k_neighbors=15, random_state=self.DETERMINISTIC_FACTOR)
    y = data.pop(class_var).values
    X = data.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = pd.concat([pd.DataFrame(smote_X), pd.DataFrame(smote_y)], axis=1)
    df_smote.columns = list(data.columns) + [class_var]

    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
    print('Minority class=', positive_class, ':', smote_target_count[positive_class])
    print('Majority class=', negative_class, ':', smote_target_count[negative_class])
    print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')

    print(df_smote[class_var].value_counts())

    return df_smote

  def explore_graph(self) -> None:
    class_var = 'class'
    target_count = self.data[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}
    
    figure()
    bar_chart(target_count.index, target_count.values, title='Class balance')
    show()