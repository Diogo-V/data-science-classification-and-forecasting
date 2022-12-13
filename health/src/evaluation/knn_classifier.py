import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from matplotlib.pyplot import figure, savefig, show, subplots, tight_layout
from ds_charts import plot_evaluation_results_2, plot_evaluation_results_2_train_test_matrixes, multiple_line_chart, plot_confusion_matrix
import math
import numpy as np

HEIGHT: int = 4

class Knn_classifier:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.train_data = train_data
        self.train_y = self.train_data.pop('readmitted').values
        self.test_data = test_data
        self.test_y = self.test_data.pop('readmitted').values
        self.neighbours = [3,5,7,9,11,13,15,17,19,21]
        self.large_neighbours = [21, 221, 421, 621, 821, 1021, 1221]
        self.dist = ['manhattan', 'euclidean', 'chebyshev']

    def explore_best_k_value(self, method="def"):
        print(self.train_data)
        print(self.test_data)
        values = {}
        best = (0, '')
        last_best = 0
        if method == "def":
            K = self.neighbours
        else:
            K = self.large_neighbours
        for dist in self.dist:
            print(f"Dist: {dist}")
            y_test_values = []
            y_train_values = []
            y_test_values_f1 = []
            y_train_values_f1 = []
            for k in K:
                print(f"K: {k}")
                knn = KNeighborsClassifier(n_neighbors=k, metric=dist)
                knn.fit(self.train_data, self.train_y)
                prd_tst_Y = knn.predict(self.test_data)
                prd_trn_Y = knn.predict(self.train_data)
                y_test_values.append(accuracy_score(self.test_y, prd_tst_Y))
                y_train_values.append(accuracy_score(self.train_y, prd_trn_Y))
                y_test_values_f1.append(f1_score(self.test_y, prd_tst_Y, average="macro"))
                y_train_values_f1.append(f1_score(self.train_y, prd_trn_Y, average="macro"))
                if y_test_values[-1] > last_best:
                    best = (k, dist)
                    last_best = y_test_values[-1]
            values[dist] = y_test_values
            self.plot_overfitting_study(K, y_train_values, y_test_values, f'KNN_{dist}_{k}','K', dist)
            self.plot_overfitting_study(K, y_train_values_f1, y_test_values_f1, f'KNN_{dist}_{k}_F1','K', dist)

        print(values)
        figure()
        figure()
        _, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
        multiple_line_chart(K, values, ax=axs[0], title='KNN variants', xlabel='n', ylabel="Accuracy Score", percentage=True)
        multiple_line_chart(K, values, ax=axs[1], title='KNN variants', xlabel='n', ylabel="F1 Score", percentage=True)
        if method == "def":
            savefig(f'health/records/evaluation/knn_k_distance_exploration.png')
        else:
            savefig(f'health/records/evaluation/knn_k_distance_exploration_{method}.png')
        print('Best results with %d neighbors and %s'%(best[0], best[1]))
        return best[0], best[1]

    def plot_overfitting_study(self, xvalues, prd_trn, prd_tst, name, xlabel, ylabel):
        evals = {'Train': prd_trn, 'Test': prd_tst}
        figure()
        multiple_line_chart(xvalues, evals, ax = None, title=f'Overfitting {name}', xlabel=xlabel, ylabel=ylabel, percentage=True)
        savefig(f'health/records/evaluation/{name}_overfitting_exploration.png')

    def compute_knn_best_results(self, k_value: int, approach: str):
        labels = pd.unique(self.test_y)
        labels.sort()

        labels_str=["1", "2", "3"]	

        knn = KNeighborsClassifier(n_neighbors=k_value, metric=approach)
        knn.fit(self.train_data, self.train_y)
        prd_trn = knn.predict(self.train_data)
        prd_tst = knn.predict(self.test_data)
        train_acc = accuracy_score(self.train_y, prd_trn)
        test_acc = accuracy_score(self.test_y, prd_tst)
        error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

        plot_evaluation_results_2_train_test_matrixes(labels, self.train_y, prd_trn, self.test_y, prd_tst)
        savefig('health/records/evaluation/knn_best_metrics.png')

        f= open('health/records/evaluation/knn_best_details.txt', 'w')
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

