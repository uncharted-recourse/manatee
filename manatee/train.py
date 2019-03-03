from Sloth.classify import Shapelets
from Sloth.preprocess import events_to_rates, events_to_densities
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
    values, counts = np.unique(y_train)
    max_samples = counts.max()
    initial = X_train.shape[0]

    for val in values:
        ixs = np.where(y_train == val)[0]
        reps = int(max_samples / len(ixs))
        X_train.append(X_train[ixs]*reps, ignore_index=True)
        print("Augmented class {} with {} repetitions (of all its samples)".format(val, reps))
    print("Before augmentation the dataset had {} samples".format(initial))
    print("After augmentation the dataset has {} samples".format(X_train.shape[0]))

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

def batch_events_to_rates(data, index, labels_dict, series_size = 60*60, min_points = 10, num_bins = 60, density=False):
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
                if density:
                    rate_vals, rate_times = events_to_densities(events.values.astype(int), num_bins = num_bins)
                else:
                    rate_vals, rate_times = events_to_rates(events.values.astype(int), num_bins = num_bins)
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
    print("\nDataset Summary: \n{} total time series, length = {} hr, sampled {} times".format(series_count, series_size, num_bins))
    for val in index.unique():
        ct = val_series_count[val]
        print("{} time series ({} %) were added from cluster: {}".format(ct, round(ct / series_count * 100, 1), val))

    labels = np.array(labels)
    series_values = np.vstack(series_values)
    series_times = np.vstack(series_times)

    # save series values and series times if they don't already exist
    if not os.path.isfile("rate_values/series_values_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points)):
        np.save("rate_values/series_values_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points), series_values)
        np.save("rate_values/series_times_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points), series_times)
        np.save("rate_values/labels_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points), labels)
        output = open("rate_values/val_series_count_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points), 'wb')
        pickle.dump(val_series_count, output)
        output.close()
    return series_values, series_times, labels, val_series_count

def train_shapelets(X_train, y_train, visualize = False, series_size = 60 * 60, num_bins = 60, density = False):

    # visualize training data
    for label in y_train:
        for index in np.where(y_train == label)[0][:2]:
            plt.scatter(np.arange(X_train.shape[1]), X_train[index])
            time_unit = series_size / num_bins / 60
            if series_size == 1:
                plt.xlabel('Minute of the Hour')
            elif series_size == 0.5:
                plt.xlabel('Half Minute of the Hour')
            if density:
                plt.ylabel('Email Density')
            else:
                plt.ylabel('Emails per Second')
            if label == 1:
                plt.title('Example of Anomalous Rate Function')
            else:
                plt.title('Example of Non-Anomalous Rate Function')
            plt.show()

    # data augmentation

    # shapelet classifier
    epochs = 100
    length = 0.1
    num_shapelet_lengths = 1
    num_shapelets = 0.1
    learning_rate = .01
    weight_regularizer = .01
    #source_dir = ''
    print("Fitting Shapelet Classifier")
    clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer)
    inds = np.arange(X_train.shape[0])
    np.random.shuffle(inds)
    X_train = X_train[inds]
    y_train = y_train[inds]
    val_split = int(0.3 * X_train.shape[0])
    model = clf.fit(X_train, y_train)
    
    # evaluate after full training
    y_pred = clf.predict(X_train[-val_split:])
    print('Evaluation on Randomly Shuffled Validation Set')
    evaluate(y_train[-val_split:], y_pred)

    # visualize 
    if visualize:
        print('Visualize All Shapelets')
        clf.VisualizeShapelets()

        print('Visualize Shapelet Classifications')
        rates = X_train[-val_split:]
        y_true = y_train[-val_split:]
        for i in np.arange(3):
            plt.scatter(np.arange(len(rates[0]), rates[i]))
            time_unit = series_size / num_bins / 60
            if series_size == 1:
                plt.xlabel('Minute of the Hour')
            elif series_size == 0.5:
                plt.xlabel('Half Minute of the Hour')
            if density:
                plt.ylabel('Email Density')
            else:
                plt.ylabel('Emails per Second')
            if y_true == 1 and y_pred == 1:
                plt.title('Correct Classification: Anomalous')
            elif y_true == 1 and y_pred == 0:
                plt.title('Incorrect Classification: True = Anomalous, Predicted = Non-Anomalous')
            elif y_true == 0 and y_pred == 1:
                plt.title('Incorrect Classification: True = Non-Anomalous, Predicted = Anomalous')
            else:
                plt.title('Correct Classification: Non-Anomalous')
            plt.show()
            clf.VisualizeShapeletLocations(rates, i)

    # hyperparameter optimization

    # shapelet sizes grid search

    # epoch optimization with best HPs and shapelet sizes
    pass

# main method for training methods
if __name__ == '__main__':
    series_size = 60 * 60
    num_bins = 60
    min_points = 10
    density = True
    data = pd.read_pickle('../../all_emails_clustered.pkl')
    data = parse_weekly_timestamps(data)        # add weekly timestamps
    index = data['file']

    if os.path.isfile("rate_values/series_values_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points)):
        series_values = np.load("rate_values/series_values_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points))
        labels = np.load("rate_values/labels_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points))
        pkl_file = open("rate_values/val_series_count_sz_{}_hr_bins_{}_min_pts_{}.npy".format(series_size / 60 / 60, num_bins, min_points), 'rb')
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
        series_values, series_times, labels, val_series_count = batch_events_to_rates(data['Weekly Timestamp'], index, labels_dict, num_bins = num_bins, density = density)
    '''
    train_shapelets(series_values.reshape(-1, series_values.shape[1], 1), labels, 
                    visualize=True, series_size = series_size, num_bins = num_bins, density=density)
    '''