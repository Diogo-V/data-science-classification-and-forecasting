import pandas as pd
from sklearn.impute import SimpleImputer
from ds_charts import get_variable_types
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



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

		self.drop_column('weight')
		self.drop_column('payer_code')
		self.fill_missing_values('medical_specialty', 'most_frequent')
		self.drop_records()

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

		knn = KNeighborsClassifier(n_neighbors=10)
		knn.fit(X_train, y_train)
		predict = knn.predict(X_test)
		result = accuracy_score(y_test, predict)
		print(result)
