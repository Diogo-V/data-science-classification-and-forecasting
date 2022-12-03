import pandas as pd
from sklearn.impute import SimpleImputer
from ds_charts import get_variable_types, plot_evaluation_results
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import math


class Inputator:

	DETERMINISM_FACTOR = 3
	# NEIGHBORS = [3, 5, 7, 9, 11, 13]
	NEIGHBORS = [7]

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
		self.evaluate_nb()

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
		self.data.to_csv(f'{self.file_out}/data_mvi_approach2.csv', index=True)
		self.data.describe(include='all')

		self.evaluate_knn()
		self.evaluate_nb()

	def drop_column(self, column_name: str):
		self.data = self.data.drop(columns=[column_name])	

	def fill_missing_values(self, column_name: str, strategy: str):
		self.data[column_name] = self.data[column_name].apply(lambda x: 0 if type(x) is not str else 1)

	def drop_records(self):
		self.data.dropna(axis=0, how='any', inplace=True)

	def evaluate_knn(self):
		best_test_acc = -1
		best_k_value = -1

		data = pd.DataFrame(self.data)
		y = data.pop('readmitted').values
		X = data.values

		# We need to create a classifier for each number of neighbors
		for n in self.NEIGHBORS:

			# Holds training and testing accuracy to be latter used to determine which K is more susceptible to over fit
			train_acc = []
			test_acc = []

			print(f"Classifying n = {n}:")

			# Creates a k fold cross validator
			skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.DETERMINISM_FACTOR)

			# Creates KNN classifier for n neighbors
			clf = KNeighborsClassifier(n, weights="uniform", p=2, metric="minkowski")

			# For each train/test set, we use a KNN classifier
			for train_index, test_index in skf.split(X, y):

				# Uses indexes to fetch which values are going to be used to train and test
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]

				# Trains knn classifier
				clf.fit(X_train, y_train.ravel())

				# Uses testing data and gets model accuracy
				acc = clf.score(X_test, y_test)
				test_acc.append(acc)

				# Uses training data and gets model accuracy to determine over fitting
				acc = clf.score(X_train, y_train)
				train_acc.append(acc)

			# Calculates means for train and test to determine which one is over fitting less
			train_mean = sum(train_acc) / 10
			test_mean = sum(test_acc) / 10
			error = math.sqrt(np.square(np.subtract(train_acc, test_acc)).mean())
			print("Training acc: {:.3f}".format(train_mean))
			print("Test acc: {:.3f}".format(test_mean))
			print("Diff between train and test: {:.3f}".format(train_mean - test_mean))
			print("Root mean squared error: {:.3f}".format(error))

			if test_mean > best_test_acc:
				best_test_acc = test_mean
				best_k_value = n

		print("MVI KNN Acc: " + best_test_acc + ", K: " + best_k_value)


		### FIXME: CANT USE THIS, NOT BINARY CLASS 
		# prd_trn = knn.predict(X_train)
		# prd_tst = knn.predict(X_test)

		# plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
		# savefig(f'{self.img_out}/knn_approach1_results.png')

	def evaluate_nb(self):

		# Holds training and testing accuracy to compute mean
		train_acc = []
		test_acc = []

		data = pd.DataFrame(self.data)
		y = data.pop('readmitted').values
		X = data.values

		
		# Creates a Gaussian Naive Bayes classifier
		gnb = GaussianNB()

		# Creates a k fold cross validator
		skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.DETERMINISM_FACTOR)


		# For each train/test set, we use a KNN classifier
		for train_index, test_index in skf.split(X, y):

			# Uses indexes to fetch which values are going to be used to train and test
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			
			labels = pd.unique(y_train)
			labels.sort()

			# Uses testing data and gets model accuracy
			acc = gnb.score(X_test, y_test)
			test_acc.append(acc)
			print("Acc using test data {:.3f}".format(acc))

			# Uses training data and gets model accuracy to determine over fitting
			acc = gnb.score(X_train, y_train)
			train_acc.append(acc)
			print("Acc using training data {:.3f}".format(acc))

		# Calculates means for train and test to determine which one is over fitting less
		train_mean = sum(train_acc) / 10
		test_mean = sum(test_acc) / 10
		error = math.sqrt(np.square(np.subtract(train_acc, test_acc)).mean())
		print("Training acc: {:.3f}".format(train_mean))
		print("Test acc: {:.3f}".format(test_mean))
		print("Diff between train and test: {:.3f}".format(train_mean - test_mean))
		print("Root mean squared error: {:.3f}".format(error))
		
		print("MVI NB Acc: " + test_mean)
		
		
		### FIXME: CANT USE THIS, NOT BINARY CLASS 
		# prd_trn = nb.predict(X_train)
		# prd_tst = nb.predict(X_test)

		# plot_evaluation_results(labels, y_train, prd_trn, y_test, prd_tst)
		# savefig(f'{self.img_out}/nb_approach1_results.png')

