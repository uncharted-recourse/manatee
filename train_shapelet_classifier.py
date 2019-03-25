import pandas as pd
import os
import numpy as np
from manatee.preprocess import parse_weekly_timestamps
from manatee.shapelet_train import train_shapelets, batch_events_to_rates
import pickle

series_size = 240 * 60
num_bins = 300
min_points = 5
filter_bandwidth = 2
density = True
data = pd.read_pickle('../all_emails_kmeans_clustered.pkl')
data = parse_weekly_timestamps(data)        # add weekly timestamps
index = data['file']

if os.path.isfile("manatee/rate_values/kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}/series_values.npy".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)):
    dir_path = "manatee/rate_values/kmeans/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)
    series_values =  np.load(dir_path + "/series_values.npy")
    # change this line from 'labels.npy' to 'labels_multi.npy' for binary vs. multiclass
    labels =  np.load(dir_path + "/labels.npy")
    '''
    pkl_file = open(dir_path + "/val_series_count.pkl", 'rb')
    val_series_count = pickle.load(pkl_file)
    pkl_file.close()
    series_count = 0
    for val in index.unique():
        series_count += val_series_count[val]
    print("\nDataset Summary: {} total time series, length = {} hr, sampled {} times\n".format(series_count, series_size / 60 / 60, num_bins))  
    for val in index.unique():
        ct = val_series_count[val]
        print("{} time series ({} % of total) were added from cluster: {}".format(ct, round(ct / series_count * 100, 1), val))
    '''
else: 
    # uncomment for binary shapelet classifier
    '''
    labels_dict = {}
    for val in data['file'].unique():
        if val == 'enron.jsonl':
            labels_dict[val] = 0
        else:
            labels_dict[val] = 1
    '''
    # uncomment for multiclass shapelet classifier
    labels_dict = {}
    for val in data['file'].unique():
        labels_dict[val] = val
    ## TODO - BATCH EVENTS TO RATES hp optimization / fidelity - series_size, num_bins, min_points, filter_bandwidth
    # train multiclass shapelet classifier without transfer learning
    series_values, series_times, labels, val_series_count = \
        batch_events_to_rates(data['Weekly Timestamp'], index, labels_dict, series_size = series_size, min_points = min_points, 
            num_bins = num_bins, filter_bandwidth = filter_bandwidth, density = density)

# randomly shuffle before splitting into training / test / val

np.random.seed(0)
randomize = np.arange(len(series_values))
np.random.shuffle(randomize)
series_values = series_values[randomize]
labels = labels[randomize]

# train
train_split = int(0.9 * series_values.shape[0])
train_shapelets(series_values[:train_split].reshape(-1, series_values.shape[1], 1), labels[:train_split])

# test eval
train_shapelets(series_values[:train_split].reshape(-1, series_values.shape[1], 1), labels[:train_split], 
    val_data = (series_values[train_split:].reshape(-1, series_values.shape[1], 1), labels[train_split:]))
# CHANGES FOR MULTICLASS

# 1. change p_threshold
# 2. uncomment target_names in shapelet_train.py
# 3. change labels to labels_multi 
# (transfer) 4. add transfer = True flag