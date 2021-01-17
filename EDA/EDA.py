import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path = r'C:\DATA\Projects\PortFolioProjects\ClassificationProject\Code'
sys.path.insert(0, path)
from UTIL import FileIO
config = FileIO.read_yaml(r'CONFIG\config_data.yml')

df = pd.read_csv(config['CSV_file_path'])
fin_data = pd.read_csv(config['fin_data_file_path'])
# https://towardsdatascience.com/violin-plots-explained-fb1d115e023d
# df.iloc[data.nlargest(10).index]['username']

# Find users with the highest number of followers.
# Sort with the number of followers:
# tmp = df.sort_values(by=['num_follower'], ascending=False)
# remove rows with repeating username:
# tmp_2 = tmp.drop_duplicates('username', inplace=False)
# tmp_3 = tmp_2.head(10)
# tmp_3.loc[:][['username','num_follower']]

# Find the tweets with the highest retweet
# tmp = df.sort_values(by=['num_retweet'], ascending=False)
# tmp.head(10)['tweet']
# tmp.head(10).iloc[:]['date_time(Local)']

fig, ax = plt.subplots(1, 1, figsize=(15, 5))

# Dist. of tweet # followers

""" data = df.loc[:]['num_follower']
fixed_bins = list(np.arange(0, 100000000, 1000000))
data_to_dis = np.clip(data, fixed_bins[0], fixed_bins[-1])
ax = sns.distplot(data_to_dis, hist=True, kde=False,
             bins=fixed_bins, color='blue',
             hist_kws={'edgecolor': 'black', 'log':True},
             ax=ax)

# End Dist. of tweet # followers
"""

"""
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
data = df.loc[:]['num_retweet']
fixed_bins = list(np.arange(0, 20000, 200))
data_to_dis = np.clip(data, fixed_bins[0], fixed_bins[-1])
ax = sns.distplot(data_to_dis, hist=True, kde=False,
             bins=fixed_bins, color='blue',
             hist_kws={'edgecolor': 'black', 'log':True},
             ax=ax)
"""

# Dist. of tweet dates
tmp = df.loc[:]['date_time(Local)']

tmp_2 = tmp.apply(lambda x: pd.Timestamp(x, tz='US/Eastern'))
data = tmp_2.apply(lambda x: (x - tmp_2[0]).days)
data = data.rename(columns={'date_time(Local)': 'days'})

fixed_bins = list(np.arange(0, 300, 1))
# data_to_dis = np.clip(data, fixed_bins[0], fixed_bins[-1])
data_to_dis = data
ax = sns.distplot(data_to_dis, hist=True, kde=False,
                  bins=fixed_bins, color='blue',
                  hist_kws={'edgecolor': 'black', 'log': True},
                  ax=ax)
# End Dist. of tweet dates

plt.show()

tmp = 1
