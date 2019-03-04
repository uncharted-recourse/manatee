from Sloth.classify import Shapelets
from Sloth.preprocess import events_to_rates
import pandas as pd
import numpy as np
import os.path
from evaluate import evaluate
from preprocess import parse_weekly_timestamps
import matplotlib.pyplot as plt
import pickle

def data_augmentation(X_train, y_train):
    '''
        augment samples of less frequent classes
        so that all classes have same number of samples
    '''
    values, counts = np.unique(y_train, return_counts = True)
    max_samples = counts.max()
    initial = X_train.shape[0]

    for val in values:
        ixs = np.where(y_train == val)[0]
        reps = int(max_samples / len(ixs)) - 1
        if reps > 0:
            X_train = np.append(X_train, X_train[ixs]*reps, axis = 0)
            y_train = np.append(y_train, y_train[ixs]*reps, axis = 0)
        print("Augmented class {} with {} repetitions".format(val, reps  * len(ixs)))
    print("\nBefore augmentation the training / validation dataset had {} samples".format(initial))
    print("After augmentation the training / validation dataset has {} samples".format(X_train.shape[0]))
    return X_train, y_train

def data_augmentation_with_noise():
    '''
        augment samples of less frequent classes (with noise)
        so that all classes have same number of samples
    '''
    pass

def shapelet_sizes_grid_search():
    '''
        grid search over different shapelet dictionary options
        from Grabocka paper
    '''
    pass

def batch_events_to_rates(data, index, labels_dict, series_size = 60*60, min_points = 10, num_bins = 60, filter_bandwidth = 1, density=False):
    '''
        convert list of event times into rate functions using a gaussian filer.

        parameters:
            data         pandas Series containing event times to convert
            index        index containing labels of different clusters to turn into 
                         unique rate functions
            labels_dict  dictionary of labels (1 per index) to apply to labels
            series_size  length of windows that time series capture, expressed in seconds (default = 24 hours)
            min_points   minimum number of points needed to calculate rate function over series of length
                         series_size
            num_bins     number of bins to subdivide series into
            filter_bandwidth       length of gaussian filter 
            density      whether to generate density rate functions

        return:
            series_values   pandas dataframe containing series values for each series
            series_times    pandas dataframe containing time values for each point in each series
    '''

    # assert that data and index have the same length
    try:
        assert(len(data) == len(index))
    except:
        raise ValueError("The series of event times and the series of indices must be the same length")

    # convert each cluster of time series to rates
    series_values = []
    series_times = []
    labels = []
    series_count = 0
    val_series_count = {}
    for val in index.unique():
        val_series_count[val] = 0
        event_times = data.loc[index == val]

        # iterate through event times by series size -> only convert to rate function if >= min_points events
        event_index = 0
        while (event_index <= event_times.max()):
            events = event_times[(event_index <= event_times) & (event_times < (event_index + series_size))]
            if len(events) >= min_points:
                rate_vals, rate_times = events_to_rates(events.values.astype(int), num_bins = num_bins, filter_bandwidth = filter_bandwidth,
                        min_time = event_index, max_time = event_index + series_size, density = density)
                series_values.append(rate_vals)
                series_times.append(rate_times)
                labels.append(labels_dict[val])
                series_count += 1
                val_series_count[val] += 1
                print("Added time series with {} events from cluster {}".format(len(events), val))
            else:
                print("Time series from cluster {} is too short".format(val))
            event_index += series_size
        print("{} time series were added from cluster: {}".format(val_series_count[val], val))
    print("\nDataset Summary: \n{} total time series, length = {} hr, sampled {} times".format(series_count, series_size / 60 / 60, num_bins))
    for val in index.unique():
        ct = val_series_count[val]
        print("{} time series ({} %) were added from cluster: {}".format(ct, round(ct / series_count * 100, 1), val))

    labels = np.array(labels)
    series_values = np.vstack(series_values)
    series_times = np.vstack(series_times)

    # save series values and series times if they don't already exist
    if not os.path.isfile("rate_values/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}/series_values.npy".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)):
        dir_path = "sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)
        os.mkdir("rate_values/" + dir_path)
        np.save("rate_values/" + dir_path + "/series_values.npy", series_values)
        np.save("rate_values/" + dir_path + "/series_times.npy", series_times)
        np.save("rate_values/" + dir_path + "/labels.npy", labels)
        output = open("rate_values/" + dir_path + "/val_series_count.pkl", 'wb')
        pickle.dump(val_series_count, output)
        output.close()
    return series_values, series_times, labels, val_series_count

def train_shapelets(X_train, y_train, visualize = False, series_size = 60 * 60, num_bins = 60, density = False):

    # visualize training data
    if visualize:
        for label in np.unique(y_train):
            for index in np.where(y_train == label)[0][:2]:
                plt.plot(np.arange(X_train.shape[1]), X_train[index])
                time_unit = series_size / num_bins / 60
                if time_unit == 1:
                    plt.xlabel('Minute of the Hour')
                elif time_unit == 0.5:
                    plt.xlabel('Half Minute of the Hour')
                if density:
                    plt.ylabel('Email Density')
                else:
                    plt.ylabel('Emails per Minute')
                if label == 1:
                    plt.title('Example of Anomalous Rate Function')
                else:
                    plt.title('Example of Non-Anomalous Rate Function')
                plt.show()
    
    # data augmentation
    X_train, y_train = data_augmentation(X_train, y_train)

    # shapelet classifier
    epochs = 100
    length = 0.1
    num_shapelet_lengths = 1
    num_shapelets = 0.1
    learning_rate = .01
    weight_regularizer = .01
    source_dir = '../shapelets'
    val_split = 1 / 3
    print("\nFitting Shapelet Classifier on {} Training Time Series".format(int((1 - val_split) * X_train.shape[0])))
    clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer)
    inds = np.arange(X_train.shape[0])
    np.random.shuffle(inds)
    X_train = X_train[inds]
    y_train = y_train[inds]
    model = clf.fit(X_train, y_train, source_dir = source_dir)
    
    # evaluate after full training
    val_split = int(val_split * X_train.shape[0])
    y_pred = clf.predict(X_train[-val_split:])
    print('\nEvaluation on Randomly Shuffled Validation Set with {} Validation Time Series'.format(val_split))
    evaluate(y_train[-val_split:], y_pred)

    # visualize 
    if visualize:
        print('Visualize All Shapelets')
        clf.VisualizeShapelets()

        print('Visualize Shapelet Classifications')
        rates = X_train[-val_split:]
        y_true = y_train[-val_split:]
        for i in np.arange(3):
            if y_true[i] == 1 and y_pred[i] == 1:
                print('Correct Classification: Anomalous')
            elif y_true[i] == 1 and y_pred[i] == 0:
                print('Incorrect Classification: True = Anomalous, Predicted = Non-Anomalous')
            elif y_true[i] == 0 and y_pred[i] == 1:
                print('Incorrect Classification: True = Non-Anomalous, Predicted = Anomalous')
            else:
                print('Correct Classification: Non-Anomalous')
            clf.VisualizeShapeletLocations(rates, i, series_size, num_bins, density)

    # Shapelet test - track over time
    track = np.array([])
    for i in range(10):
        track = np.append(track, i * 0.01)
    track = np.append(track, track[::-1])

    # Beginning
    track_beg = np.pad(track, (0, num_bins - len(track)), 'constant')
    track_beg = track_beg.reshape(1, track_beg.shape[0], 1)
    track_pred = clf.predict(track_beg)
    if track_pred:
        print('Classification: Anomalous')
    else:
        print('Classification: Non-Anomalous')
    clf.VisualizeShapeletLocations(track_beg, 0, series_size, num_bins, density)

    # Middle
    l = int((num_bins - len(track)) / 2)
    track_mid = np.pad(track, (l, l), 'constant')
    track_mid = track_mid.reshape(1, track_mid.shape[0], 1)
    track_pred = clf.predict(track_mid)
    if track_pred:
        print('Classification: Anomalous')
    else:
        print('Classification: Non-Anomalous')
    clf.VisualizeShapeletLocations(track_mid, 0, series_size, num_bins, density)

    # End
    track_end = np.pad(track, (num_bins - len(track), 0), 'constant')
    track_end = track_end.reshape(1, track_end.shape[0], 1)
    track_pred = clf.predict(track_end)
    if track_pred:
        print('Classification: Anomalous')
    else:
        print('Classification: Non-Anomalous')
    clf.VisualizeShapeletLocations(track_end, 0, series_size, num_bins, density)

    clf.clear_session()
    # hyperparameter optimization

    # shapelet sizes grid search

    # epoch optimization with best HPs and shapelet sizes

# main method for training methods
if __name__ == '__main__':
    series_size = 60 * 60
    num_bins = 60
    min_points = 10
    filter_bandwidth = 1
    density = True
    data = pd.read_pickle('../../all_emails_clustered.pkl')
    data = parse_weekly_timestamps(data)        # add weekly timestamps
    index = data['file']

    if os.path.isfile("rate_values/sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}/series_values.npy".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)):
        dir_path = "sz_{}_hr_bins_{}_min_pts_{}_filter_width_{}_density_{}".format(series_size / 60 / 60, num_bins, min_points, filter_bandwidth, density)
        series_values =  np.load("rate_values/" + dir_path + "/series_values.npy")
        labels =  np.load("rate_values/" + dir_path + "/labels.npy")
        pkl_file = open("rate_values/" + dir_path + "/val_series_count.pkl", 'rb')
        val_series_count = pickle.load(pkl_file)
        pkl_file.close()
        series_count = 0
        for val in index.unique():
            series_count += val_series_count[val]
        print("\nDataset Summary: {} total time series, length = {} hr, sampled {} times\n".format(series_count, series_size / 60 / 60, num_bins))  
        for val in index.unique():
            ct = val_series_count[val]
            print("{} time series ({} % of total) were added from cluster: {}".format(ct, round(ct / series_count * 100, 1), val))
    else:
        labels_dict = {}
        for val in data['file'].unique():
            if val == 'enron.jsonl':
                labels_dict[val] = 0
            else:
                labels_dict[val] = 1
        ## TODO - BATCH EVENTS TO RATES hp optimization / fidelity - series_size, num_bins, min_points, filter_bandwidth
        series_values, series_times, labels, val_series_count = \
            batch_events_to_rates(data['Weekly Timestamp'], index, labels_dict, series_size = series_size, min_points = min_points, 
                num_bins = num_bins, filter_bandwidth = filter_bandwidth, density = density)
    train_split = int(0.9 * series_values.shape[0])
    train_shapelets(series_values[:train_split].reshape(-1, series_values.shape[1], 1), labels[:train_split], 
                    visualize=False, series_size = series_size, num_bins = num_bins, density=density)
    