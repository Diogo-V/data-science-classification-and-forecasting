import pandas as pd
from collections.abc import Callable

class Parser:

  def __init__(self, data: pd.DataFrame, missing_values_str: str) -> None:
    """
    Description:
      * Builds a parser that transforms symbolic variables into numeric ones.

    Arguments:
      * data(pd.DataFrame): data to to be transformed
      * missing_values_str(str): representation of missing values in dataset
    """
    self.data: pd.DataFrame = data
    self.missing: str = missing_values_str
    self.missing_parsed: int = -1

    # Holds mapping between columns and functions used to transform them
    self.func_map: dict[str, Callable[[str], int]] = {
      'race': self._map_race,
      'gender': self._map_gender,
      'age': self._map_age,
      'diag_1': self._map_diag, 
      'diag_2': self._map_diag,
      'diag_3': self._map_diag,
      'max_glu_serum': self._map_max_glu_serum,
      'A1Cresult': self._map_a1c_result,
      'metformin': self._map_medication,
      'repaglinide': self._map_medication, 
      'nateglinide': self._map_medication, 
      'chlorpropamide': self._map_medication, 
      'glimepiride': self._map_medication,
      'glipizide': self._map_medication, 
      'glyburide': self._map_medication, 
      'pioglitazone': self._map_medication,
      'rosiglitazone': self._map_medication, 
      'acarbose': self._map_medication, 
      'miglitol': self._map_medication, 
      'tolazamide': self._map_medication, 
      'examide': self._map_medication, 
      'citoglipton': self._map_medication, 
      'insulin': self._map_medication, 
      'glyburide-metformin': self._map_medication,
      'readmitted': self._map_readmitted
    }

  def parse_dataset(self, output_path: str) -> None:
    """
    Description:
      * Uses input dataset to parse symbolic column values into numeric ones.

    Arguments:
      * output_path(str): output dataset path to be stored in
    """
    for column, parse_func in self.func_map.items():
      print(f"Parsing column: [{column}]...")
      self.data[column] = self.data[column].apply(lambda x: self.missing_parsed if x == self.missing else parse_func(x))
    print(f"Storing resulting dataset in [{output_path}]...")
    print("DONE :)")
    self.data.to_csv(output_path)

  def _map_race(self, ele: str) -> int:
    """
    Description:
      * Builds and applies a map that can convert a string element from race column into a number.

    Arguments:
      * ele(str): column value

    Returns:
      * int: corresponding value
    """
    mapper: dict[str, int] = {
      'AfricanAmerican': 1,
      'Asian': 2,
      'Caucasian': 3,
      'Hispanic': 4,
      'Other': 5,
      'Unknown': 6,
    }
    return mapper[ele]

  def _map_gender(self, ele: str) -> int:
    """
    Description:
      * Builds and applies a map that can convert a string element from gender column into a number.

    Arguments:
      * ele(str): column value

    Returns:
      * int: corresponding value
    """
    mapper: dict[str, int] = {
      'Female': 1,
      'Male': 2,
      'Unknown/Invalid': 3,
    }
    return mapper[ele]
  
  def _map_age(self, ele: str) -> int:
    """
    Description:
      * Builds and applies a map that can convert a string element from age column into a number.

    Arguments:
      * ele(str): column value

    Returns:
      * int: corresponding value
    """
    mapper: dict[str, int] = {
      '[0-10)': 1,
      '[10-20)': 2,
      '[20-30)': 3,
      '[30-40)': 4,
      '[40-50)': 5,
      '[50-60)': 6,
      '[60-70)': 7,
      '[70-80)': 8,
      '[80-90)': 9,
      '[90-100)': 10,
    }
    return mapper[ele]
  
  def _map_diag(self, ele: str) -> int:
    """
    Description:
      * Builds and applies a map that can convert a string element from diag_1, diag_2, diag_3 columns into a number.

    Arguments:
      * ele(str): column value

    Returns:
      * int: corresponding value
    """
    mapper: dict[int, int] = {
      800: 8,
      780: 8,
      760: 7,
      740: 6,
      710: 5,
      680: 4,
      630: 3,
      580: 2,
      520: 1,
      460: 8,
      390: 7,
      320: 6,
      290: 5,
      280: 4,
      240: 3,
      140: 2,
      1: 1,
    }

    try:
      
      # Evaluates element and converts into a number to check boundaries
      result = eval(ele)

      # Gets keys from mapper and since they are reversed (highest > lowest), we don't need to check boundaries
      for boundary, value in mapper.items():
        if result >= boundary:
          return value
    
    except:
      return 8  # If eval throws an error, we know its a string, and thus is number 8


  def _map_max_glu_serum(self, ele: str) -> int:
    """
    Description:
      * Builds and applies a map that can convert a string element from max_glu_serum column into a number.

    Arguments:
      * ele(str): column value

    Returns:
      * int: corresponding value
    """
    mapper: dict[str, int] = {
      '>300': 4,
      '>200': 3,
      'Norm': 2,
      'None': 1
    }
    return mapper[ele]

  def _map_a1c_result(self, ele: str) -> int:
    """
    Description:
      * Builds and applies a map that can convert a string element from a1c_result column into a number.

    Arguments:
      * ele(str): column value

    Returns:
      * int: corresponding value
    """
    mapper: dict[str, int] = {
      '>7': 3,
      '>8': 4,
      'Norm': 2,
      'None': 1
    }
    return mapper[ele]

  def _map_medication(self, ele: str) -> int:
    """
    Description:
      * Builds and applies a map that can convert a string element from medication columns into a number.

    Arguments:
      * ele(str): column value

    Returns:
      * int: corresponding value
    """
    mapper: dict[str, int] = {
      'Down': 2,
      'Up': 4,
      'Steady': 3,
      'No': 1
    }
    return mapper[ele]

  def _map_readmitted(self, ele: str) -> int:
    """
    Description:
      * Builds and applies a map that can convert a string element from readmitted column into a number.

    Arguments:
      * ele(str): column value

    Returns:
      * int: corresponding value
    """
    mapper: dict[str, int] = {
      '<30': 3,
      '>30': 2,
      'NO': 1,
    }
    return mapper[ele]