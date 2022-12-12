import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.pyplot import figure, savefig, show, subplots, tight_layout
from ds_charts import plot_evaluation_results, multiple_line_chart, plot_overfitting_study, plot_confusion_matrix

class Knn_classifier:

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.train_data = train_data
        self.train_y = self.train_data.pop('readmitted').values
        self.test_data = test_data
        self.test_y = self.test_data.pop('readmitted').values
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
        savefig(f'health/records/evaluation/knn_k_distance_exploration.png')
        print('Best results with %d neighbors and %s'%(best[0], best[1]))
        return best[0], best[1]

    def compute_knn_best_results(self, k_value: int, approach: str):
        labels = pd.unique(self.test_y)
        labels.sort()

        labels_str=["1", "2", "3"]	

        knn = KNeighborsClassifier(n_neighbors=k_value, metric=approach)
        knn.fit(self.train_data, self.train_y)
        predict = knn.predict(self.test_data)
        result = accuracy_score(self.test_y, predict)
        print('Accuracy:', result)
        
        figure()
        fig, axs = subplots(1, 2, figsize=(8, 4), squeeze=False)
        plot_confusion_matrix(confusion_matrix(self.test_y, predict, labels=labels), labels, ax=axs[0,0], )
        plot_confusion_matrix(confusion_matrix(self.test_y, predict, labels=labels), labels, ax=axs[0,1], normalize=True)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[0,0].text(0.05,0.95,f'Best approach: K = {k_value} with {approach} metric', transform=axs[0,0].transAxes, position=(0, 1.2), fontsize=12, verticalalignment='top', bbox=props)
        tight_layout()
        savefig(f'health/records/evaluation/knn_k_distance_best_results.png')
