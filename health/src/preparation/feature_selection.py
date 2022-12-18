import pandas as pd
from matplotlib.pyplot import figure, title, savefig, show
from seaborn import heatmap
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart


class FeatureSelection:

    def __init__(self,data: pd.DataFrame) -> None:
        self.data = data
        self.threshold = 0.9
    
    def explore_redundat(self): 
        drop, corr_mtx = self.select_redundant(self.data.corr())
        # print(drop.keys())

        if corr_mtx.empty:
            raise ValueError('Matrix is empty.')

        figure(figsize=[10, 10])
        heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
        title('Filtered Correlation Analysis')
        savefig(f'health/records/preparation/filtered_correlation_analysis_{self.threshold}.png')

        return drop

    def select_redundant(self, corr_mtx):
        if corr_mtx.empty:
            return {}

        corr_mtx = abs(corr_mtx)
        vars_2drop = {}
        for el in corr_mtx.columns:
            el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= self.threshold]
            if len(el_corr) == 1:
                corr_mtx.drop(labels=el, axis=1, inplace=True)
                corr_mtx.drop(labels=el, axis=0, inplace=True)
            else:
                vars_2drop[el] = el_corr.index
        return vars_2drop, corr_mtx

    def drop_redundant(self, vars_2drop: dict) -> pd.DataFrame:
        sel_2drop = []
        print(vars_2drop.keys())
        for key in vars_2drop.keys():
            if key not in sel_2drop:
                for r in vars_2drop[key]:
                    if r != key and r not in sel_2drop:
                        sel_2drop.append(r)
        print('Variables to drop', sel_2drop)
        df = self.data.copy()
        for var in sel_2drop:
            df.drop(labels=var, axis=1, inplace=True)
        return df

    
    def select_low_variance(self, data: pd.DataFrame):
        numeric_vars = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_diagnoses', 'number_inpatient']

        data_numeric = data[numeric_vars]

        lst_variables = []
        lst_variances = []
        for el in data_numeric:
            value = self.data[el].var()
            if value <= self.threshold:
                lst_variables.append(el)
                lst_variances.append(value)

        figure(figsize=[10, 10])
        bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance', rotation=True)
        savefig('health/records/preparation/filtered_variance_analysis.png')


    
    