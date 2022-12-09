import pandas as pd
from collections.abc import Callable

class Parser:

	def __init__(self, data: pd.DataFrame) -> None:
		"""
		Description:
			* Builds a parser that transforms symbolic variables into numeric ones.

		Arguments:
			* data(pd.DataFrame): data to to be transformed
			* missing_values_str(str): representation of missing values in dataset
		"""
		self.data: pd.DataFrame = data

	def parse_dataset(self, output_path: str) -> pd.DataFrame:
		dates = self.data.pop('date') 
		
		days = []
		months = []
		years = []

		for date in dates:
			list = date.split("/")
			days.append(list[0])
			months.append(list[1])
			years.append(list[2])

		self.data.insert(loc=len(self.data.columns), column='day', value=days)
		self.data.insert(loc=len(self.data.columns), column='month', value=months)
		self.data.insert(loc=len(self.data.columns), column='year', value=years)
		
		return self.data