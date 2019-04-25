# -*- coding: utf-8 -*-
'''
@Time    : 2019/4/25 21:01
@Author  : Zekun Cai
@File    : pid_group.py
@Software: PyCharm
'''
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np

data = pd.read_pickle('../../data/processed/new_process/train_agg_0.04.pkl')
pid_data = pd.read_csv('../../data/profiles.csv')
data.pid.fillna(0, inplace=True)
data = data[data.click_mode.notnull()]
pid_group = pd.DataFrame()

for i in range(12):
    pid_group['pid_prop' + str(i)] = data.groupby('pid').apply(lambda x: sum(x.click_mode == i) / len(x))

pid_group.to_csv('../../data/processed/new_process/pid_distribute.csv')
pid_group.reset_index(inplace=True)

pid_all = pd.merge(pid_data[['pid']], pid_group, how='outer')
pid_all.fillna(1 / 12, inplace=True)
pid_all.set_index('pid', inplace=True)

bandwidth = estimate_bandwidth(pid_all.values, quantile=0.2, n_samples=25000, n_jobs=-1, random_state=123)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True, n_jobs=-1)
ms.fit(pid_all.values)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

pid_all['pid_label'] = labels
pid_all.to_csv('../../data/processed/new_process/pid_label' + str(n_clusters_) + '.csv')
