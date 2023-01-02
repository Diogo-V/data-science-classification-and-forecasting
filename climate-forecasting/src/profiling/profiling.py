from matplotlib import legend
import pandas as pd
from matplotlib.pyplot import figure, xticks, savefig, subplots, show
from ts_functions import plot_series_multivariate, HEIGHT

class Profiler:

  def __init__(self, data: pd.DataFrame) -> None:
    self.data = data

    index = self.data.index.to_period('W')
    self.week_df = self.data.copy().groupby(index).mean()
    self.week_df['date'] = index.drop_duplicates().to_timestamp()
    self.week_df.set_index('date', drop=True, inplace=True)

    index = self.data.index.to_period('M')
    self.month_df = self.data.copy().groupby(index).mean()
    self.month_df['date'] = index.drop_duplicates().to_timestamp()
    self.month_df.set_index('date', drop=True, inplace=True)

    index = self.data.index.to_period('Q')
    self.quarter_df = self.data.copy().groupby(index).mean()
    self.quarter_df['date'] = index.drop_duplicates().to_timestamp()
    self.quarter_df.set_index('date', drop=True, inplace=True)


  def explore_dimensionality(self):
    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    plot_series_multivariate(self.data, x_label='date', y_label='values', title='DROUGHT')
    xticks(rotation = 45)
    savefig('climate-forecasting/records/profiling/distribution.png')

    f = open('climate-forecasting/records/profiling/distribution_details.txt', 'w')
    f.write(f"Nr. Records = {self.data.shape[0]}\n")
    f.write(f"First Date = {self.data.index[0]}\n")
    f.write(f"Last Date = {self.data.index[-1]}\n")
    f.close()


  def explore_granularity(self):

    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    plot_series_multivariate(self.data, title='Daily values', x_label='date', y_label='consumption')
    xticks(rotation = 45)
    savefig('climate-forecasting/records/profiling/granularity_day.png')

    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    plot_series_multivariate(self.week_df, title='Weekly values', x_label='date', y_label='consumption')
    xticks(rotation = 45)
    savefig('climate-forecasting/records/profiling/granularity_week.png')
    
    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series_multivariate(self.month_df, title='Monthly values', x_label='date', y_label='consumption')
    savefig('climate-forecasting/records/profiling/granularity_month.png')

    figure(figsize=(3*HEIGHT, HEIGHT))
    plot_series_multivariate(self.quarter_df, title='Quarterly values', x_label='date', y_label='consumption')
    savefig('climate-forecasting/records/profiling/granularity_quaterly.png')

 
  """
   Description:
    Analizes distribution through the 5-number summary and boxplots. It is doing for the most atomic time series (daily)
  """ 
  def explore_distribution_boxplots(self):
    f = open('climate-forecasting/records/profiling/distribution_boxplot_daily_details.txt', 'w')
    f.write(f"Daily\n")
    f.write(str(self.data.describe()))
    f.close()


    figure(figsize=(3*HEIGHT, 3*HEIGHT))
    self.data.boxplot()
    savefig('climate-forecasting/records/profiling/distribution_boxplot_daily.png')


  def explore_distribution_histograms(self):
    labels = ["PRECTOT", "PS", "T2M", "T2MDEW", "T2MWET", "TS", "QV2M"]

    granularity = ['daily', 'weekly', 'monthly']

    data = [self.data, self.week_df, self.month_df]

    for g in granularity:
        index = granularity.index(g)
        bins = (10, 25, 50)
        _, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
        for j in range(len(bins)):
            axs[j].set_title(f'Histogram for {g} {bins[j]} bins')
            axs[j].set_xlabel('consumption')
            axs[j].set_ylabel('Nr records')
            axs[j].hist(data[index].values, bins=bins[j], label=labels)
            axs[j].legend(loc="upper right", prop={'size': 5})
    
        savefig(f'climate-forecasting/records/profiling/distribution_histograms_{g}.png')