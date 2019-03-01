from Sloth.classify import Shapelets
from Sloth.preprocess import events_to_rates
import pandas as pd
import numpy as np

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

def batch_events_to_rates(data, index = None, series_size = 60*60*24, min_points = 10, num_bins = 72):
    '''
        convert list of event times into rate functions using a gaussian filer.

        parameters:
            data         pandas Series containing event times to convert
            index        optional index containing labels of different clusters to turn into 
                         unique rate functions
            series_size  length of windows that time series capture, expressed in seconds (default = 24 hours)
            min_points   minimum number of points needed to calculate rate function over series of length
                         series_size
            num_bins     number of bins to subdivide series into

        return:
            series_values   pandas dataframe containing series values for each series
            series_times    pandas dataframe containing time values for each point in each series
    '''

    # assert that data and index have the same length
    if index is not None:
        try:
            assert(len(data) == len(index))
        except:
            raise ValueError("The series of event times and the series of indices must be the same length")

    # convert each cluster of time series to rates
    series_values = []
    series_times = []
    series_count = 0
    val_series_count = {}
    for val in index.unique():
        val_series_count[val] = 0
        event_times = data.loc[index == val]

        # iterate through event times by series size -> only convert to rate function if >= min_points events
        event_index = event_times.min()
        while (event_index + series_size <= event_times.max()):
            events = [event_times[(event_index <= event_times) & (event_times < (event_index + series_size))]]
            if len(events[0]) >= min_points:
                rate_vals, rate_times = events_to_rates(events, num_bins = num_bins)
                series_values.append(rate_vals)
                series_times.append(rate_times)
                series_count += 1
                val_series_count[val] += 1
                print("Added time series with {} events from cluster {}".format(len(events[0]), val))
            else:
                print("Time series from cluster {} is too short".format(val))
            event_index += series_size
        print("{} time series were added from cluster: {}".format(val_series_count[val], val))
    
    print("\nSummary: \n{} total time series were added".format(series_count))
    for val in index.unique():
        ct = val_series_count[val]
        print("{} time series ({} %) were added from cluster: {}".format(ct, round(ct / series_count * 100, 2), val))

    return pd.DataFrame(np.vstack(series_values)), pd.DataFrame(np.vstack(series_times))

def train_shapelets(X_train, y_train):

    # data augmentation

    # shapelet classifier
    epochs = 100
    length = 0.1
    num_shapelet_lengths = 2
    num_shapelets = 0.2
    learning_rate = .01
    weight_regularizer = .01
    source_dir = ''
    train_split = 0.7
    print("Fitting Shapelet Classifier")
    clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer)
    clf.fit(X_train, y_train, source_dir, train_split)

    # experiment with different ways to parse rate function

    # hyperparameter optimization

    # shapelet sizes grid search

    # epoch optimization with best HPs and shapelet sizes
    pass


# main method for training methods
if __name__ == '__main__':
    data = pd.read_pickle('../../all_emails_clustered.pkl')
    series_values, series_times = batch_events_to_rates(data['Timestamp'], data['file'])
    labels = np.array((data['file'] != 'enron.jsonl').astype(int))
    train_shapelets(series_values.values.reshape(-1, series_values.shape[1], 1), labels)
