import pandas as pd
from sklearn.impute import SimpleImputer
from ds_charts import plot_confusion_matrix, plot_evaluation_results_2
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report


class MVImputation:

	DETERMINISM_FACTOR = 3
	NEIGHBORS = [3, 5, 7, 9, 11, 13]

	def __init__(self, data: pd.DataFrame, missing_values_str: str) -> None:
		"""
		Description:
		* Class to transform the original data, dealing with missing values

		Arguments:
		* data(pd.DataFrame): data to to be transformed
		* missing_values_str(str): representation of missing values in dataset
		"""

		self.data: pd.DataFrame = data
		# self.data.replace({missing_values_str, np.nan}, regex=True, inplace=True)
		for c in self.data:
			self.data[c] = self.data[c].map(lambda x: np.nan if x == missing_values_str or x == -1 else x)

	def compute_mv_imputation(self, file_out_path: str) -> pd.DataFrame:
		return self.approach_2(file_out_path)

	def approach_1(self, file_out_path: str) -> pd.DataFrame:
		"""
		- Drop column Weight
		- Drop column Payer_Code
		- Drop column Medical Speciality
		- Drop all other records with missing values (cincluding race and gender that have enconding for missing value)

		- Evaluate with KNN
		- Evaluate with NB
		"""

		self.drop_column('weight')
		self.drop_column('payer_code')
		self.drop_column('medical_specialty')
		self.drop_records()

		self.data.to_csv(f'{file_out_path}/data_mvi_approach1.csv')

		self.evaluate_knn('approach_1')
		self.evaluate_nb('approach_1')

		return self.data

	def approach_2(self, file_out_path: str):
		"""
		- Drop column Weight
		- Drop column Payer_Code
		- Drop column Medical Speciality
		- Substitute missing values with mean/most frequent value value

		- Evaluate with KNN
		- Evaluate with NB
		"""

		self.file_out = file_out_path

		self.drop_column('weight')
		self.drop_column('payer_code')
		self.drop_column('medical_specialty')
		
		tmp_nr, tmp_sb, tmp_bool = None, None, None

		numeric_vars = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_diagnoses', 'number_inpatient']
		symbolic_vars = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'readmitted', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'acetohexamide', 'troglitazone', 'tolbutamide']
		binary_vars = ['diabetesMed', 'change']

		tmp_nr, tmp_sb, tmp_bool = None, None, None
		if len(numeric_vars) > 0:
			imp = SimpleImputer(strategy='mean', missing_values=np.nan, copy=True)
			tmp_nr = pd.DataFrame(imp.fit_transform(self.data[numeric_vars]), columns=numeric_vars)
		if len(symbolic_vars) > 0:
			imp = SimpleImputer(strategy='most_frequent', missing_values=np.nan, copy=True)
			tmp_sb = pd.DataFrame(imp.fit_transform(self.data[symbolic_vars]), columns=symbolic_vars)
		if len(binary_vars) > 0:
			imp = SimpleImputer(strategy='most_frequent', missing_values=np.nan, copy=True)
			tmp_bool = pd.DataFrame(imp.fit_transform(self.data[binary_vars]), columns=binary_vars)

		self.data = pd.concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
		self.data.index = self.data.index
		self.data.to_csv(f'{self.file_out}/data_mvi_approach2.csv', index=True)
		self.data.describe(include='all')

		self.evaluate_knn('approach_2')
		self.evaluate_nb('approach_2')

		return self.data

	def drop_column(self, column_name: str):
		self.data = self.data.drop(columns=[column_name])	

	def fill_missing_values(self, column_name: str, strategy: str):
		self.data[column_name] = self.data[column_name].apply(lambda x: 0 if type(x) is not str else 1)

	def drop_records(self):
		self.data.dropna(axis=0, how='any', inplace=True)

	def evaluate_knn(self, approach: str):

		data = pd.DataFrame(self.data)
		y = data.pop('readmitted').values
		X = data.values

		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=self.DETERMINISM_FACTOR)

		labels = pd.unique(y)
		labels.sort()

		labels_str=["1", "2", "3"]	

		knn = KNeighborsClassifier(n_neighbors=13)
		knn.fit(X_train, y_train)
		prd_train = knn.predict(X_train)
		prd_tst = knn.predict(X_test)
		train_acc = accuracy_score(y_train, prd_train)
		test_acc = accuracy_score(y_test, prd_tst)
		error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

		plot_evaluation_results_2(labels, y_train, prd_train, y_test, prd_tst)
		plt.savefig(f'health/records/preparation/mvi_{approach}_knn.png')

		f= open(f'health/records/preparation/mvi_{approach}_knn_details.txt', 'w')
		f.write("Accuracy Train: {:.5f}\n".format(train_acc))
		f.write("Accuracy Test: {:.5f}\n".format(test_acc))
		f.write("Diff between train and test: {:.5f}\n".format(train_acc - test_acc))
		f.write("Root mean squared error: {:.5f}\n".format(error))
		f.write("########################\n")
		f.write("Train\n")
		f.write(classification_report(y_train, prd_train,target_names=labels_str))
		f.write("Test\n")
		f.write(classification_report(y_test, prd_tst,target_names=labels_str))


	def evaluate_nb(self, approach: str):

		data = pd.DataFrame(self.data)
		y = data.pop('readmitted').values
		X = data.values

		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=self.DETERMINISM_FACTOR)

		labels = pd.unique(y)
		labels.sort()

		labels_str=["1", "2", "3"]	

		nb = GaussianNB()	
		nb.fit(X_train, y_train)
		prd_train = nb.predict(X_train)
		prd_tst = nb.predict(X_test)
		train_acc = accuracy_score(y_train, prd_train)
		test_acc = accuracy_score(y_test, prd_tst)
		error = math.sqrt(np.square(np.subtract(train_acc, test_acc)) / 2)

		plot_evaluation_results_2(labels, y_train, prd_train, y_test, prd_tst)
		plt.savefig(f'health/records/preparation/mvi_{approach}_nb.png')

		f= open(f'health/records/preparation/mvi_{approach}_nb_details.txt', 'w')
		f.write("Accuracy Train: {:.5f}\n".format(train_acc))
		f.write("Accuracy Test: {:.5f}\n".format(test_acc))
		f.write("Diff between train and test: {:.5f}\n".format(train_acc - test_acc))
		f.write("Root mean squared error: {:.5f}\n".format(error))
		f.write("########################\n")
		f.write("Train\n")
		f.write(classification_report(y_train, prd_train,target_names=labels_str))
		f.write("Test\n")
		f.write(classification_report(y_test, prd_tst,target_names=labels_str))
		

