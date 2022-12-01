import pandas as pd
from sklearn.impute import SimpleImputer
from ds_charts import get_variable_types, plot_evaluation_results
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



class Inputator:

	def __init__(self, data: pd.DataFrame, missing_values_str: str) -> None:
		"""
		Description:
		* Class to transform the original data, dealing with missing values

		Arguments:
		* data(pd.DataFrame): data to to be transformed
		* missing_values_str(str): representation of missing values in dataset
		"""

		self.data: pd.DataFrame = data
		# self.data.replace({missing_values_str, nan}, regex=True, inplace=True)
		for c in self.data:
			self.data[c] = self.data[c].map(lambda x: nan if x == missing_values_str or x == -1 else x)

	def approach_1(self, img_out_path: str, file_out_path: str):
		"""
		- Drop column Weight
		- Drop column Payer_Code
		- Split Medical Specialty in binary: No (0) or Yes (1)
		- Drop all other records with missing values

		- Evaluate with KNN
		- Evaluate with NB
		"""

		self.img_out = img_out_path
		self.file_out = file_out_path

		self.drop_column('weight')
		self.drop_column('payer_code')
		self.fill_missing_values('medical_specialty', 'most_frequent')
		self.drop_records()

		self.data.to_csv(f'{self.file_out}/data_mvi_approach1.csv')

		self.evaluate_knn()


	def drop_column(self, column_name: str):
		self.data = self.data.drop(columns=[column_name])	

	def fill_missing_values(self, column_name: str, strategy: str):
		self.data[column_name] = self.data[column_name].apply(lambda x: 0 if type(x) is not str else 1)

	def drop_records(self):
		self.data.dropna(axis=0, how='any', inplace=True)

	def evaluate_knn(self):
		data = pd.DataFrame(self.data)
		y = data.pop('readmitted').values
		X = data.values

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

		labels = pd.unique(y_train)
		labels.sort()

		knn = KNeighborsClassifier(n_neighbors=10)
		knn.fit(X_train, y_train)
		predict = knn.predict(X_test)
		result = accuracy_score(y_test, predict)
		print(result)

		### FIXME: CANT USE THIS, NOT BINARY CLASS 
		# prd_trn = knn.predict(X_train)
		# prd_tst = knn.predict(X_test)

		# plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
		# savefig(f'{self.img_out}/knn_approach1_results.png')

	def evaluate_nb(self):
		data = pd.DataFrame(self.data)
		y = data.pop('readmitted').values
		X = data.values

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

		labels = pd.unique(y_train)
		labels.sort()