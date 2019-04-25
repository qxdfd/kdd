# -*- coding: utf-8 -*-
'''
@Time    : 2019/4/25 11:24
@Author  : Zekun Cai
@File    : merge_file.py
@Software: PyCharm
'''
import pandas as pd
import numpy as np
import time
from chinese_calendar import is_holiday
import json
import warnings

warnings.filterwarnings('ignore')


def process_query(name):
    print('process query:', name)
    if name == 'train':
        train = pd.read_csv(data_path + 'train_queries.csv')
    else:
        train = pd.read_csv(data_path + 'test_queries.csv')
    train['o_lon'] = train.o.apply(lambda x: float(x.split(',')[0]))
    train['o_lat'] = train.o.apply(lambda x: float(x.split(',')[1]))
    train['d_lon'] = train.d.apply(lambda x: float(x.split(',')[0]))
    train['d_lat'] = train.d.apply(lambda x: float(x.split(',')[1]))

    train['o_i'] = (train['o_lon'] - lon_min) // precision
    train['o_j'] = (train['o_lat'] - lat_min) // precision
    train['d_i'] = (train['d_lon'] - lon_min) // precision
    train['d_j'] = (train['d_lat'] - lat_min) // precision

    num_i = np.ceil((lon_max - lon_min) / precision)
    num_j = np.ceil((lat_max - lat_min) / precision)

    train['o_grid'] = train['o_i'] * num_j + train['o_j']
    train['d_grid'] = train['d_i'] * num_j + train['d_j']

    train['req_time'] = pd.to_datetime(train['req_time'])
    train['hod'] = train['req_time'].dt.hour
    train['dow'] = train['req_time'].dt.dayofweek + 1
    train['holiday'] = train.req_time.apply(lambda x: is_holiday(x) * 1)
    train['holiday'][(train.dow == 6) | (train.dow == 7)] = 1

    if name == 'train':
        train.to_csv(save_path + 'train_od_date.csv', index=False)
    else:
        train.to_csv(save_path + 'test_od_date.csv', index=False)
    train.drop(['o', 'd', 'req_time'], axis=1, inplace=True)

    return train


def order_n(x, num):
    while True:
        try:
            return x[num]
        except:
            num -= 1


def min_nozero(x):
    try:
        return np.min(x[np.nonzero(x)])
    except:
        return 0


def min_nozero_loc(x):
    try:
        return np.where(x == np.min(x[np.nonzero(x)]))[0][0]
    except:
        return 0


def process_plan(name):
    print('process plan:', name)
    if name == 'train':
        plan = pd.read_csv(data_path + 'train_plans.csv')
    else:
        plan = pd.read_csv(data_path + 'test_plans.csv')
    plan['plans'] = plan.apply(lambda x: json.loads(x[2]), axis=1)
    new_plan = pd.concat([plan['sid'], plan.plans.apply(pd.Series)], axis=1)
    new_plan.set_index('sid', inplace=True)
    new_plan = new_plan.stack().reset_index('sid')
    new_plan.reset_index(inplace=True, drop=True)
    new_plan.columns = ['sid', 'plans']

    new_plan['distance'] = new_plan.plans.apply(lambda x: x['distance'])
    new_plan['price'] = new_plan.plans.apply(lambda x: x['price'])
    new_plan['eta'] = new_plan.plans.apply(lambda x: x['eta'])
    new_plan['transport_mode'] = new_plan.plans.apply(lambda x: x['transport_mode'])

    new_plan['price'][new_plan.price == ''] = 0
    new_plan['price'] = new_plan.price.astype(int)

    new_plan[['sid', 'distance', 'price', 'eta', 'transport_mode']].to_csv(save_path +
                                                                           'test_plan_processed.csv', index=False)

    agg_format = pd.DataFrame()
    agg_format['agg_dis'] = new_plan.groupby('sid').apply(lambda x: x.distance.values)
    agg_format['agg_price'] = new_plan.groupby('sid').apply(lambda x: x.price.values)
    agg_format['agg_eta'] = new_plan.groupby('sid').apply(lambda x: x.eta.values)
    agg_format['agg_mode'] = new_plan.groupby('sid').apply(lambda x: x.transport_mode.values)
    agg_format = agg_format.reset_index()

    agg_format['mode_num'] = agg_format.agg_mode.apply(len)
    agg_format['first_mode'] = agg_format.agg_mode.apply(lambda x: x[0])
    agg_format['second_mode'] = agg_format.agg_mode.apply(order_n, args=(1,))
    agg_format['third_mode'] = agg_format.agg_mode.apply(order_n, args=(2,))
    agg_format['fourth_mode'] = agg_format.agg_mode.apply(order_n, args=(3,))
    agg_format['fifth_mode'] = agg_format.agg_mode.apply(order_n, args=(4,))
    agg_format['last_mode'] = agg_format.agg_mode.apply(lambda x: x[-1])

    agg_format['min_dis'] = agg_format.agg_dis.apply(np.min)
    agg_format['max_dis'] = agg_format.agg_dis.apply(np.max)
    agg_format['min_dist_loc'] = agg_format.agg_dis.apply(np.argmin)
    agg_format['max_dist_loc'] = agg_format.agg_dis.apply(np.argmax)

    # some mode distance is wrong
    agg_format['min_dist_loc'][(agg_format['min_dis'] == 1) & (agg_format['max_dis'] >= 100)] = 0
    agg_format['min_dis'] = agg_format.apply(
        lambda x: x.min_dis if (x.min_dis > 1 or x.max_dis < 100) else x.max_dis, axis=1)

    agg_format['min_dist_mode'] = agg_format.apply(lambda x: x.agg_mode[x.min_dist_loc], axis=1)
    agg_format['max_dist_mode'] = agg_format.apply(lambda x: x.agg_mode[x.max_dist_loc], axis=1)
    agg_format['first_dis'] = agg_format.agg_dis.apply(lambda x: x[0])
    agg_format['first_dis_diff_min'] = (agg_format['first_dis'] - agg_format['min_dis']) / agg_format['min_dis']
    agg_format['first_dis_diff_max'] = (agg_format['first_dis'] - agg_format['max_dis']) / agg_format['max_dis']
    agg_format['second_dis'] = agg_format.agg_dis.apply(order_n, args=(1,))
    agg_format['second_dis_diff_min'] = (agg_format['second_dis'] - agg_format['min_dis']) / agg_format['min_dis']
    agg_format['second_dis_diff_max'] = (agg_format['second_dis'] - agg_format['max_dis']) / agg_format['max_dis']
    agg_format['third_dis'] = agg_format.agg_dis.apply(order_n, args=(2,))
    agg_format['third_dis_diff_min'] = (agg_format['third_dis'] - agg_format['min_dis']) / agg_format['min_dis']
    agg_format['third_dis_diff_max'] = (agg_format['third_dis'] - agg_format['min_dis']) / agg_format['min_dis']
    agg_format['fourth_dis'] = agg_format.agg_dis.apply(order_n, args=(3,))
    agg_format['fourth_dis_diff_min'] = (agg_format['fourth_dis'] - agg_format['min_dis']) / agg_format['min_dis']
    agg_format['fourth_dis_diff_max'] = (agg_format['fourth_dis'] - agg_format['min_dis']) / agg_format['min_dis']
    agg_format['fifth_dis'] = agg_format.agg_dis.apply(order_n, args=(4,))
    agg_format['fifth_dis_diff_min'] = (agg_format['fifth_dis'] - agg_format['min_dis']) / agg_format['min_dis']
    agg_format['fifth_dis_diff_max'] = (agg_format['fifth_dis'] - agg_format['min_dis']) / agg_format['min_dis']
    agg_format['mean_dis'] = agg_format.agg_dis.apply(np.mean)
    agg_format['std_dis'] = agg_format.agg_dis.apply(np.var)

    # agg_format['min_price'] = agg_format.agg_price.apply(np.min)
    agg_format['min_nozero_price'] = agg_format.agg_price.apply(min_nozero)
    agg_format['max_price'] = agg_format.agg_price.apply(np.max)
    agg_format['min_nozero_price_loc'] = agg_format.agg_price.apply(min_nozero_loc)
    agg_format['max_price_loc'] = agg_format.agg_price.apply(np.argmax)
    agg_format['min_nozero_price_mode'] = agg_format.apply(lambda x: x.agg_mode[x.min_nozero_price_loc], axis=1)
    agg_format['max_price_mode'] = agg_format.apply(lambda x: x.agg_mode[x.max_price_loc], axis=1)
    agg_format['first_price'] = agg_format.agg_price.apply(lambda x: x[0])
    agg_format['first_price_diff_min'] = \
        (agg_format['first_price'] - agg_format['min_nozero_price']) / (agg_format['min_nozero_price'] + 1)
    agg_format['first_price_diff_max'] = \
        (agg_format['first_price'] - agg_format['max_price']) / (agg_format['max_price'] + 1)
    agg_format['second_price'] = agg_format.agg_price.apply(order_n, args=(1,))
    agg_format['second_price_diff_min'] = \
        (agg_format['second_price'] - agg_format['min_nozero_price']) / (agg_format['min_nozero_price'] + 1)
    agg_format['second_price_diff_max'] = \
        (agg_format['second_price'] - agg_format['max_price']) / (agg_format['max_price'] + 1)
    agg_format['third_price'] = agg_format.agg_price.apply(order_n, args=(2,))
    agg_format['third_price_diff_min'] = \
        (agg_format['third_price'] - agg_format['min_nozero_price']) / (agg_format['min_nozero_price'] + 1)
    agg_format['third_price_diff_max'] = \
        (agg_format['third_price'] - agg_format['max_price']) / (agg_format['max_price'] + 1)
    agg_format['fourth_price'] = agg_format.agg_price.apply(order_n, args=(3,))
    agg_format['fourth_price_diff_min'] = \
        (agg_format['fourth_price'] - agg_format['min_nozero_price']) / (agg_format['min_nozero_price'] + 1)
    agg_format['fourth_price_diff_max'] = \
        (agg_format['fourth_price'] - agg_format['max_price']) / (agg_format['max_price'] + 1)
    agg_format['fifth_price'] = agg_format.agg_price.apply(order_n, args=(4,))
    agg_format['fifth_price_diff_min'] = \
        (agg_format['fifth_price'] - agg_format['min_nozero_price']) / (agg_format['min_nozero_price'] + 1)
    agg_format['fifth_price_diff_max'] = \
        (agg_format['fifth_price'] - agg_format['max_price']) / (agg_format['max_price'] + 1)
    agg_format['mean_price'] = agg_format.agg_price.apply(np.mean)
    agg_format['std_price'] = agg_format.agg_price.apply(np.var)

    agg_format['min_eta'] = agg_format.agg_eta.apply(np.min)
    agg_format['max_eta'] = agg_format.agg_eta.apply(np.max)
    agg_format['min_eta_loc'] = agg_format.agg_eta.apply(np.argmin)
    agg_format['max_eta_loc'] = agg_format.agg_eta.apply(np.argmax)
    agg_format['min_eta_mode'] = agg_format.apply(lambda x: x.agg_mode[x.min_eta_loc], axis=1)
    agg_format['max_eta_mode'] = agg_format.apply(lambda x: x.agg_mode[x.max_eta_loc], axis=1)
    agg_format['first_eta'] = agg_format.agg_eta.apply(lambda x: x[0])
    agg_format['first_eta_diff_min'] = (agg_format['first_eta'] - agg_format['min_eta']) / agg_format['min_eta']
    agg_format['first_eta_diff_max'] = (agg_format['first_eta'] - agg_format['max_eta']) / agg_format['max_eta']
    agg_format['second_eta'] = agg_format.agg_eta.apply(order_n, args=(1,))
    agg_format['second_eta_diff_min'] = (agg_format['second_eta'] - agg_format['min_eta']) / agg_format['min_eta']
    agg_format['second_eta_diff_max'] = (agg_format['second_eta'] - agg_format['max_eta']) / agg_format['max_eta']
    agg_format['third_eta'] = agg_format.agg_eta.apply(order_n, args=(2,))
    agg_format['third_eta_diff_min'] = (agg_format['third_eta'] - agg_format['min_eta']) / agg_format['min_eta']
    agg_format['third_eta_diff_max'] = (agg_format['third_eta'] - agg_format['max_eta']) / agg_format['max_eta']
    agg_format['fourth_eta'] = agg_format.agg_eta.apply(order_n, args=(3,))
    agg_format['fourth_eta_diff_min'] = (agg_format['fourth_eta'] - agg_format['min_eta']) / agg_format['min_eta']
    agg_format['fourth_eta_diff_max'] = (agg_format['fourth_eta'] - agg_format['max_eta']) / agg_format['max_eta']
    agg_format['fifth_eta'] = agg_format.agg_eta.apply(order_n, args=(4,))
    agg_format['fifth_eta_diff_min'] = (agg_format['fifth_eta'] - agg_format['min_eta']) / agg_format['min_eta']
    agg_format['fifth_eta_diff_max'] = (agg_format['fifth_eta'] - agg_format['max_eta']) / agg_format['max_eta']
    agg_format['mean_eta'] = agg_format.agg_eta.apply(np.mean)
    agg_format['std_eta'] = agg_format.agg_eta.apply(np.var)

    if name == 'train':
        agg_format.to_pickle(save_path + 'train_plan_feature.pkl')
    else:
        agg_format.to_pickle(save_path + 'test_plan_feature.pkl')
    agg_format.drop(['agg_dis', 'agg_price', 'agg_eta', 'agg_mode'], inplace=True, axis=1)
    return agg_format


def build_statistical_feature():
    print('build stat')
    train = pd.read_csv(save_path + 'train_od_date.csv')
    plan = pd.read_csv(data_path + 'train_plans.csv')
    click = pd.read_csv(data_path + 'train_clicks.csv')
    plan = pd.merge(plan, click[['sid', 'click_mode']], how='left')
    plan.fillna(0, inplace=True)
    train = pd.merge(train, plan[['sid', 'click_mode']], how='right')

    o_distribute, d_distribute, hod_distribute = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(12):
        o_distribute['o_grid_prop' + str(i)] = train.groupby('o_grid').apply(lambda x: sum(x.click_mode == i) / len(x))
    for i in range(12):
        d_distribute['d_grid_prop' + str(i)] = train.groupby('d_grid').apply(lambda x: sum(x.click_mode == i) / len(x))
    for i in range(12):
        hod_distribute['hod_prop' + str(i)] = train.groupby('hod').apply(lambda x: sum(x.click_mode == i) / len(x))

    o_distribute.to_csv(save_path + 'o_distribute.csv')
    d_distribute.to_csv(save_path + 'd_distribute.csv')
    hod_distribute.to_csv(save_path + 'hod_distribute.csv')

    o_distribute.reset_index(inplace=True)
    d_distribute.reset_index(inplace=True)
    hod_distribute.reset_index(inplace=True)

    return o_distribute, d_distribute, hod_distribute


def read_statistical_feature():
    o_distribute = pd.read_csv(save_path + 'o_distribute.csv')
    d_distribute = pd.read_csv(save_path + 'd_distribute.csv')
    hod_distribute = pd.read_csv(save_path + 'hod_distribute.csv')
    return o_distribute, d_distribute, hod_distribute


def merge_file(query, plan, o_sata, d_stat, hod_stat, name):
    print('merge file:', name)
    data = pd.merge(query, plan, on='sid', how='left')
    data = pd.merge(data, o_sata, on='o_grid', how='left')
    data = pd.merge(data, d_stat, on='d_grid', how='left')
    data = pd.merge(data, hod_stat, on='hod', how='left')
    if name == 'train':
        click = pd.read_csv(data_path + 'train_clicks.csv')
        tmp = pd.merge(plan[['sid']], click[['sid', 'click_mode']], on='sid', how='left')
        tmp.fillna(0, inplace=True)
        data = pd.merge(data, tmp, on='sid', how='left')
        data.to_pickle(save_path + 'train_agg_' + str(precision) + '.pkl')
    else:
        column = []
        for i in range(12):
            column.append('o_grid_prop' + str(i))
            column.append('d_grid_prop' + str(i))
            column.append('hod_prop' + str(i))
        data[column].fillna(1 / 12, inplace=True)
        data.to_pickle(save_path + 'test_agg_' + str(precision) + '.pkl')


def process_train():
    query = process_query('train')
    plan = process_plan('train')
    o_sata, d_stat, hod_stat = build_statistical_feature()
    merge_file(query, plan, o_sata, d_stat, hod_stat, 'train')


def prcess_test():
    query = process_query('test')
    plan = process_plan('test')
    o_sata, d_stat, hod_stat = read_statistical_feature()
    merge_file(query, plan, o_sata, d_stat, hod_stat, 'test')


data_path = '../../data/'
save_path = '../../data/processed/new_process/'
lat_max = 40.97
lat_min = 39.46
lon_max = 117.48
lon_min = 115.4
precision = 0.04

if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    process_train()
    prcess_test()

    # agg_format['first_dis_clip']
    # agg_format['first_price_clip']
    # agg_format['first_eta_clip']
    # agg_format['min_dis_clip']
    # agg_format['min_price_clip']
    # agg_format['min_eta_clip']
    # agg_format['max_dis_clip']
    # agg_format['max_price_clip']
    # agg_format['max_eta_clip']
    #
    # agg_format['o_group']
    # agg_format['d_group']
    # agg_format['pid_group']
