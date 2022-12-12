import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.pyplot import figure, savefig
from ds_charts import plot_evaluation_results, multiple_line_chart
import math
import numpy as np

class Knn_classifier:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.train_data = train_data
        self.train_y = self.train_data.pop('class').values
        self.test_data = test_data
        self.test_y = self.test_data.pop('class').values
        self.neighbours = [3,5,7,9,11,13,15]
        self.dist = ['manhattan', 'euclidean', 'chebyshev']

    def explore_best_k_value(self):
        print(self.train_data)
        print(self.test_data)
        values = {}
        best = (0, '')
        last_best = 0
        for dist in self.dist:
            y_test_values = []
            for k in self.neighbours:
                knn = KNeighborsClassifier(n_neighbors=k, metric=dist)
                knn.fit(self.train_data, self.train_y)
                prd_tst_Y = knn.predict(self.test_data)
                y_test_values.append(accuracy_score(self.test_y, prd_tst_Y))
                if y_test_values[-1] > last_best:
                    best = (k, dist)
                    last_best = y_test_values[-1]
            values[dist] = y_test_values

        print(values)
        figure()
        multiple_line_chart(self.neighbours, values, title='KNN variants', xlabel='n', ylabel="Accuracy Score", percentage=True)
        savefig(f'climate/records/evaluation/knn_k_distance_exploration.png')
        print('Best results with %d neighbors and %s'%(best[0], best[1]))
        return best[0], best[1]

    def compute_knn_best_results(self, k_value: int, approach: str):
        labels = pd.unique(self.test_y)
        labels.sort()

        labels_str=["Class 1", "Class 2"]	

        knn = KNeighborsClassifier(n_neighbors=k_value, metric=approach)
        knn.fit(self.train_data, self.train_y)
        prd_trn = knn.predict(self.train_data)
        prd_tst = knn.predict(self.test_data)
        train_acc = accuracy_score(self.train_y, prd_trn)
        test_acc = accuracy_score(self.test_y, prd_tst)
        error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

        plot_evaluation_results(labels, self.train_y, prd_trn, self.test_y, prd_tst)
        savefig('climate/records/evaluation/knn_k_distance_best_results.png')

        f= open('climate/records/evaluation/knn_k_distance_best_results_details.txt', 'w')
        f.write(f'Best approach: K = {k_value} with {approach} metric\n')
        f.write("Accuracy Train: {:.5f}\n".format(train_acc))
        f.write("Accuracy Test: {:.5f}\n".format(test_acc))
        f.write("Diff between train and test: {:.5f}\n".format(train_acc - test_acc))
        f.write("Root mean squared error: {:.5f}\n".format(error))
        f.write("########################\n")
        f.write("Train\n")
        f.write(classification_report(self.train_y, prd_trn,target_names=labels_str))
        f.write("Test\n")
        f.write(classification_report(self.test_y, prd_tst,target_names=labels_str))

