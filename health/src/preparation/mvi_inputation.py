import pandas as pd
from sklearn.impute import SimpleImputer
from ds_charts import plot_confusion_matrix
import matplotlib.pyplot as plt
from numpy import nan
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
		# self.data.replace({missing_values_str, nan}, regex=True, inplace=True)
		for c in self.data:
			self.data[c] = self.data[c].map(lambda x: nan if x == missing_values_str or x == -1 else x)

	def compute_mv_imputation(self) -> pd.DataFrame:
		self.drop_column('weight')
		self.drop_column('payer_code')
		self.drop_column('medical_specialty')
		
		tmp_nr, tmp_sb, tmp_bool = None, None, None
		## FIXME: since we enconde the data first, there is no way that this can know what variables are symbolic if they are also int
		numeric_vars = ['encounter_id', 'patient_nbr', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_diagnoses', 'number_inpatient']
		symbolic_vars = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'readmitted', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'acetohexamide', 'troglitazone', 'tolbutamide']
		binary_vars = ['diabetesMed', 'change']

		tmp_nr, tmp_sb, tmp_bool = None, None, None
		if len(numeric_vars) > 0:
			imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
			tmp_nr = pd.DataFrame(imp.fit_transform(self.data[numeric_vars]), columns=numeric_vars)
		if len(symbolic_vars) > 0:
			imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
			tmp_sb = pd.DataFrame(imp.fit_transform(self.data[symbolic_vars]), columns=symbolic_vars)
		if len(binary_vars) > 0:
			imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
			tmp_bool = pd.DataFrame(imp.fit_transform(self.data[binary_vars]), columns=binary_vars)

		self.data = pd.concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
		self.data.index = self.data.index
		return self.data

	def approach_1(self, img_out_path: str, file_out_path: str):
		"""
		- Drop column Weight
		- Drop column Payer_Code
		- Split Medical Specialty in binary: No (0) or Yes (1)
		- Drop all other records with missing values (cincluding race and gender that have enconding for missing value)

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

		self.evaluate_knn('approach1')
		self.evaluate_nb('approach1')

	def approach_2(self, img_out_path: str, file_out_path: str):
		"""
		- Drop column Weight
		- Drop column Payer_Code
		- Drop column Medical Speciality
		- Substitute missing values with mode/mean value

		- Evaluate with KNN
		- Evaluate with NB
		"""

		self.img_out = img_out_path
		self.file_out = file_out_path

		self.drop_column('weight')
		self.drop_column('payer_code')
		self.drop_column('medical_specialty')
		
		tmp_nr, tmp_sb, tmp_bool = None, None, None
		## FIXME: since we enconde the data first, there is no way that this can know what variables are symbolic if they are also int
		numeric_vars = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_diagnoses', 'number_inpatient']
		symbolic_vars = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'readmitted', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'acetohexamide', 'troglitazone', 'tolbutamide']
		binary_vars = ['diabetesMed', 'change']

		tmp_nr, tmp_sb, tmp_bool = None, None, None
		if len(numeric_vars) > 0:
			imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
			tmp_nr = pd.DataFrame(imp.fit_transform(self.data[numeric_vars]), columns=numeric_vars)
		if len(symbolic_vars) > 0:
			imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
			tmp_sb = pd.DataFrame(imp.fit_transform(self.data[symbolic_vars]), columns=symbolic_vars)
		if len(binary_vars) > 0:
			imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
			tmp_bool = pd.DataFrame(imp.fit_transform(self.data[binary_vars]), columns=binary_vars)

		self.data = pd.concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
		self.data.index = self.data.index
		self.data.to_csv(f'{self.file_out}/data_mvi_approach2.csv', index=True)
		self.data.describe(include='all')

		self.evaluate_knn('approach2')
		self.evaluate_nb('approach2')

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

		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

		labels = pd.unique(y)
		labels.sort()

		labels_str=["1", "2", "3"]	

		knn = KNeighborsClassifier(n_neighbors=13)
		knn.fit(X_train, y_train)
		predict = knn.predict(X_test)
		result = accuracy_score(y_test, predict)
		print('Accuracy:', result)

		plt.figure()
		fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
		plot_confusion_matrix(confusion_matrix(y_test, predict, labels=labels), labels, ax=axs[0,0], )
		plot_confusion_matrix(confusion_matrix(y_test, predict, labels=labels), labels, ax=axs[0,1], normalize=True)
		plt.tight_layout()
		plt.savefig(f'health/records/preparation/mvi_{approach}_knn.png')

		print(classification_report(y_test, predict,target_names=labels_str))

	def evaluate_nb(self, approach: str):

		data = pd.DataFrame(self.data)
		y = data.pop('readmitted').values
		X = data.values

		X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

		labels = pd.unique(y)
		labels.sort()

		labels_str=["1", "2", "3"]	

		nb = GaussianNB()	
		nb.fit(X_train, y_train)
		predict = nb.predict(X_test)
		result = accuracy_score(y_test, predict)
		print('Accuracy:', result)

		plt.figure()
		fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
		plot_confusion_matrix(confusion_matrix(y_test, predict, labels=labels), labels, ax=axs[0,0], )
		plot_confusion_matrix(confusion_matrix(y_test, predict, labels=labels), labels, ax=axs[0,1], normalize=True)
		plt.tight_layout()
		plt.savefig(f'health/records/preparation/mvi_{approach}_nb.png')

		print(classification_report(y_test, predict,target_names=labels_str))
		

