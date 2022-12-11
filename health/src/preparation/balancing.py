import pandas as pd
from matplotlib.pyplot import figure, show
from ds_charts import bar_chart
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import ClusterCentroids
import matplotlib.pyplot as plt

class Balancing:
  
  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data
    self.DETERMINISTIC_FACTOR = 3

  def compute_balancing(self) -> pd.DataFrame:
    self.data = self.explore_balancing_smote(self.data)
    return self.data

  def _get_symbolic_index_array(self) -> list[int]:
    headers = ['diabetesMed', 'change', 'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'A1Cresult', 'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'troglitazone', 'tolbutamide', 'metformin-rosiglitazone']
    result = []
    for h in headers:
      result.append(self.data.columns.get_loc(h))
    return result

  def explore_balancing_smote(self, data: pd.DataFrame) -> pd.DataFrame:
    class_var = 'readmitted'
    target_count = data[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    #ind_positive_class = target_count.index.get_loc(positive_class)
    print('Minority class=', positive_class, ':', target_count[positive_class])
    print('Majority class=', negative_class, ':', target_count[negative_class])
    print('Proportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    smote = SMOTENC(self._get_symbolic_index_array(), sampling_strategy='auto', k_neighbors=15, random_state=self.DETERMINISTIC_FACTOR)
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

  def explore_balancing_under_sampling(self, data: pd.DataFrame) -> pd.DataFrame:
    X = data.drop("readmitted", axis=1)
    y = data["readmitted"]
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, y)
    df = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)], axis=1)
    return df

  def explore_scatter_plot(self, data: pd.DataFrame) -> None:
    x = data['diabetesMed'].to_numpy()
    y = data['insulin'].to_numpy()
    c = data['readmitted'].to_numpy()

    plt.scatter(x, y, c=c, cmap='viridis')
    plt.show()

  def explore_graph(self) -> None:
    class_var = 'readmitted'
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