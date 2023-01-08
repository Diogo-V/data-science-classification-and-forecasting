from matplotlib import legend
import pandas as pd
from matplotlib.pyplot import figure, xticks, savefig, subplots, show
from ts_functions import plot_series, HEIGHT, plot_series
from numpy import ones

class Profiler:

  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data

    index = self.data.index.to_period('D')
    self.day_df_mean = self.data.copy().groupby(index).mean()
    self.day_df_mean['date'] = index.drop_duplicates().to_timestamp()
    self.day_df_mean.set_index('date', drop=True, inplace=True)

    self.day_df_sum = self.data.copy().groupby(index).sum()
    self.day_df_sum['date'] = index.drop_duplicates().to_timestamp()
    self.day_df_sum.set_index('date', drop=True, inplace=True)

    index = self.data.index.to_period('W')
    self.week_df_mean = self.data.copy().groupby(index).mean()
    self.week_df_mean['date'] = index.drop_duplicates().to_timestamp()
    self.week_df_mean.set_index('date', drop=True, inplace=True)

    self.week_df_sum = self.data.copy().groupby(index).sum()
    self.week_df_sum['date'] = index.drop_duplicates().to_timestamp()
    self.week_df_sum.set_index('date', drop=True, inplace=True)

    index = self.data.index.to_period('M')
    self.month_df_mean = self.data.copy().groupby(index).mean()
    self.month_df_mean['date'] = index.drop_duplicates().to_timestamp()
    self.month_df_mean.set_index('date', drop=True, inplace=True)

    self.month_df_sum = self.data.copy().groupby(index).sum()
    self.month_df_sum['date'] = index.drop_duplicates().to_timestamp()
    self.month_df_sum.set_index('date', drop=True, inplace=True)

    index = self.data.index.to_period('Q')
    self.quarter_df_mean = self.data.copy().groupby(index).mean()
    self.quarter_df_mean['date'] = index.drop_duplicates().to_timestamp()
    self.quarter_df_mean.set_index('date', drop=True, inplace=True)

    self.quarter_df_sum = self.data.copy().groupby(index).sum()
    self.quarter_df_sum['date'] = index.drop_duplicates().to_timestamp()
    self.quarter_df_sum.set_index('date', drop=True, inplace=True)


  def explore_dimensionality(self):
    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    plot_series(self.data, x_label='date', title='QV2M')
    xticks(rotation = 45)
    savefig('climate-forecasting/records/profiling/distribution.png')

    f = open('climate-forecasting/records/profiling/distribution_details.txt', 'w')
    f.write(f"Nr. Records = {self.data.shape[0]}\n")
    f.write(f"First Date = {self.data.index[0]}\n")
    f.write(f"Last Date = {self.data.index[-1]}\n")
    f.close()


  def explore_granularity(self):
    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    plot_series(self.data, title='Hourly values', x_label='date', y_label="QV2M")
    xticks(rotation = 45)
    savefig('climate-forecasting/records/profiling/granularity_hourly.png')

    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    plot_series(self.day_df_mean, title='Daily values', x_label='date', y_label="QV2M")
    xticks(rotation = 45)
    savefig('climate-forecasting/records/profiling/granularity_day.png')

    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    plot_series(self.week_df_mean, title='Weekly values', x_label='date', y_label="QV2M")
    xticks(rotation = 45)
    savefig('climate-forecasting/records/profiling/granularity_week.png')
    
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(self.month_df_mean, title='Monthly values', x_label='date', y_label="QV2M")
    savefig('climate-forecasting/records/profiling/granularity_month.png')

    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(self.quarter_df_mean, title='Quarterly values', x_label='date', y_label="QV2M")
    savefig('climate-forecasting/records/profiling/granularity_quaterly.png')


  def explore_count_data_types(self) -> None:
    """
    Description:
      * Shows counts of variables per column and missing values percentage.
    """
    for col in self.data:
      print(f"Col name: {col}")
      if self.data[col].isnull().sum() != 0:
        print(f"Percentage of missing values (blank): {round(self.data[col].isnull().sum() / self.data[col].sum() * 100, 2)}%")
      series = self.data[col].value_counts(ascending=True)
      print(series)
      dc = series.to_dict()
      if dc.get('?') is not None:
        print(f"Percentage of missing values: {round(dc['?'] / sum(dc.values()) * 100, 2)}%")
      print("###################################")
 
  """
   Description:
    Analizes distribution through the 5-number summary and boxplots. It is doing for the most atomic time series (daily)
  """ 
  def explore_distribution_boxplots(self):
    f = open('climate-forecasting/records/profiling/distribution_boxplot_hourly_details.txt', 'w')
    f.write(f"Hourly\n")
    f.write(str(self.data.describe()))
    f.close()


    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    self.data.boxplot()
    savefig('climate-forecasting/records/profiling/distribution_boxplot_hourly.png')


  def explore_distribution_histograms(self):

    granularity = ['hourly', 'daily', 'weekly', 'monthly']

    data = [self.data, self.day_df_sum, self.week_df_sum, self.month_df_sum]

    for g in granularity:
        index = granularity.index(g)
        bins = (10, 25, 50)
        _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
        for j in range(len(bins)):
            axs[j].set_title(f'Histogram for {g} QV2M {bins[j]} bins')
            axs[j].set_ylabel('Nr records')
            axs[j].set_xlabel('QV2M')
            axs[j].hist(data[index].values, bins=bins[j])
    
        savefig(f'climate-forecasting/records/profiling/distribution_histograms_{g}.png')

  def explore_stationary(self):
    BINS = 10
    line = []

    dt_series = pd.Series(self.data['QV2M'])

    mean_line = pd.Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
    series = {'ashrae': dt_series, 'mean': mean_line}
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(series, x_label='timestamp',  title='Stationary study', y_label='QV2M', show_std=True)
    savefig(f'climate-forecasting/records/profiling/stationary_fixed.png')

    n = len(dt_series)
    for i in range(BINS):
        b = dt_series[i*n//BINS:(i+1)*n//BINS]
        mean = [b.mean()] * (n//BINS)
        line += mean
    line += [line[-1]] * (n - len(line))
    mean_line = pd.Series(line, index=dt_series.index)
    series = {'QV2M': dt_series, 'mean': mean_line}
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series(series, x_label='timestamp', title='Stationary study', show_std=True, y_label="QV2M")
    
    savefig(f'climate-forecasting/records/profiling/stationary.png')
