import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sb
from sklearn.decomposition import PCA
#import umap

plt.style.use("seaborn-whitegrid")        

# NK brand colors
COLORS = ["#FA5655", "#252B7A"]

# load simon data
df = pd.read_pickle('../all_emails_clustered.pkl')

# just take Enron / 419 emails
enron_features = np.array((df.loc[(df['file'] == 'enron.jsonl')]['Simon Features']).values.tolist())
nigerian_features = np.array((df.loc[(df['file'] == 'nigerian.jsonl')]['Simon Features']).values.tolist())
enron_labels = np.array((df[(df['file'] == 'enron.jsonl')]['file']))
nigerian_labels = np.array((df[(df['file'] == 'nigerian.jsonl')]['file']))

np.random.seed(0)
randomize = np.arange(enron_features.shape[0])
np.random.shuffle(randomize)
enron_features = enron_features[randomize]

simon_features = np.append(enron_features[:10*nigerian_features.shape[0]], nigerian_features, axis = 0)
labels = np.append(enron_labels[:10*nigerian_labels.shape[0]], nigerian_labels)
labels[labels == 'enron.jsonl'] = COLORS[0]
labels[labels == 'nigerian.jsonl'] = COLORS[1]

# fit tSNE clustering
perplex = [2000,3000]
for p in perplex:
    embedded = TSNE(n_components=2, perplexity=p, n_iter=1000, init='pca').fit_transform(simon_features)

    # plot embeddings
    enron_ix = np.where(labels == COLORS[0])
    nigerian_ix = np.where(labels == COLORS[1])
    plt.clf()
    plt.scatter(embedded[enron_ix][:,0], embedded[enron_ix][:,1], c = labels[enron_ix], alpha = 0.3, label = 'Enron Emails')
    plt.scatter(embedded[nigerian_ix][:,0], embedded[nigerian_ix][:,1], c = labels[nigerian_ix], alpha = 0.3, label = '419 Spam Emails')
    leg = plt.legend(loc = 4, frameon = True)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    '''
    labels = [item.get_text() for item in plt.gca().get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    plt.gca().set_xticklabels(empty_string_labels)
    labels = [item.get_text() for item in plt.gca().get_yticklabels()]
    empty_string_labels = ['']*len(labels)
    plt.gca().set_yticklabels(empty_string_labels)
    '''
    plt.savefig("visualizations/tSNE_simon_embedding_{}_perplexity".format(p))
'''
# try PCA visualziation
pca = PCA(n_components=2)
X_r = pca.fit(simon_features).transform(simon_features)
plt.scatter(X_r[:,0], X_r[:,1], c = labels)
plt.legend(loc = 'best')
plt.show()

embedding = umap.UMAP().fit_transform(simon_features)
plt.scatter(embedding[:,0], embedding[:,1], c = labels)
'''