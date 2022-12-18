import pandas as pd
from matplotlib.pyplot import figure, show
from ds_charts import bar_chart
from imblearn.over_sampling import SMOTENC

class Balancing:
  
  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data
    self.DETERMINISTIC_FACTOR = 3

  def compute_balancing(self) -> pd.DataFrame:
    self.data = self.explore_balancing(self.data)
    return self.data

  def get_symbolic_index_array(self):
    headers = ['fips', 'SQ1', 'SQ3', 'SQ4']
    result = []
    for h in headers:
      result.append(self.data.columns.get_loc(h))
    return result

  def explore_balancing(self, data: pd.DataFrame) -> pd.DataFrame:
    class_var = 'class'
    target_count = data[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    smote = SMOTENC(self.get_symbolic_index_array(), sampling_strategy='auto', k_neighbors=15, random_state=self.DETERMINISTIC_FACTOR)
    y = data.pop(class_var).values
    X = data.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = pd.concat([pd.DataFrame(smote_X), pd.DataFrame(smote_y)], axis=1)
    df_smote.columns = list(data.columns) + [class_var]

    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
    print('Minority class=', positive_class, ':', smote_target_count[positive_class])
    print('Majority class=', negative_class, ':', smote_target_count[negative_class])
    print('Proportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1')

    print(df_smote[class_var].value_counts())

    return df_smote

  def explore_graph(self) -> None:
    class_var = 'class'
    target_count = self.data[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}
    
    figure()
    bar_chart(target_count.index, target_count.values, title='Class balance')
    show()