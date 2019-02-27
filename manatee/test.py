from robust_rcf import robust_rcf
import numpy as np
import pandas as pd

def test_rrcf(datapath):
    # load / prepare data
    data = np.load(datapath)['all_emails']
    df = pd.DataFrame(data, columns = ['Timestamp','Year', 'Month', 'Day of Month', 'Day of Week', 'Hour', 'Minute', 'Seconds', 'Simon Features', 'file'])
    # sort values by timestamp for training / testing
    df = df.sort_values(by = ['Timestamp'])
    simon_features = pd.DataFrame(df['Simon Features'].values.tolist())
    labels = (df['file'] != 'enron.jsonl').astype(int)

    # break into train / test by oldest / newest
    train_split = int(0.6 * df.shape[0])
    val_split = int(0.3 * df.shape[0])
    simon_train, y_train = simon_features[:train_split], labels[:train_split]
    simon_val, y_val = simon_features[train_split:train_split + val_split], labels[train_split:train_split + val_split]
    simon_test, y_test = simon_features[train_split + val_split:], labels[train_split + val_split:]

    # print anomalous percentage in train / val / test
    print('There are {} ({} %) anomalous examples in the train set'.format(sum(y_train), len(y_train)))
    print('There are {} ({} %) anomalous examples in the validation set'.format(sum(y_val), len(y_val)))
    print('There are {} ({} %) anomalous examples in the test set'.format(sum(y_test), len(y_test)))

    # test batch anomaly detection on SIMON features
    # initially set num_samples_per_tree based on ratio of anomalies
    tree_size = sum(labels) / (df.shape[0] - sum(labels))
    num_trees = 100

# main method for testing methods
if __name__ == '__main__':
    datapath = 'all_emails_parsed.npz'
    test_rrcf(datapath)