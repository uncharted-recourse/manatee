'''
Implementation of the Robust Random Cut Forest Algorithm for anomaly detection by Guha et al. (2016).

S. Guha, N. Mishra, G. Roy, & O. Schrijvers. Robust random cut forest based anomaly
detection on streams, in Proceedings of the 33rd International conference on machine
learning, New York, NY, 2016 (pp. 2712-2721).

Python implementation from https://github.com/kLabUM/rrcf
'''

import numpy as np 
import pandas as pd
import rrcf as rrcf_base
from evaluate import evaluate
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance

class robust_rcf():
    '''
        This class creates a robust random cut forest for anomlay detection

        Hyperparameters:
            num_trees: the number of trees in the random forest
            tree_size: the number of points sampled for each tree (according to AWS, tree_size should be chosen such that 
                       1/tree_size approximates the ratio of anomalous data to normal data)
    '''
    def __init__(self, num_trees, tree_size):

        # assert that num_trees and tree_size are passed as hyperparameters
        try:
            assert(num_trees is not None)
        except:
            raise ValueError("Must pass num_trees (the number of trees in the forest) as a hyperparameter")
        try:
            assert(num_trees is not None)
        except:
            raise ValueError("Must pass tree_size (the number of points in each tree) as a hyperparameter")

        self.num_trees = num_trees
        self.tree_size = tree_size
        self.num_points = None
        self.forest = None
        self.dimension = None
    
    def fit_batch(self, points):
        '''
            Creates a rrcf with num_trees trees from random samples of size tree_size from a batch set of points

            Parameters:
                points:   the points from which to create the rrcf. np.ndarray of size (n x d)
        '''

        # assert that points.shape has two dimensions
        try:
            assert(len(points.shape) is 2)
        except:
            raise ValueError("Input points must have shape (n x d)")
        self.num_points = points.shape[0]
        self.dimension = points.shape[1]
        
        # create forest
        forest = []

        # scale points between 0 and 1 ??? might improve
        #points = TimeSeriesScalerMinMax().fit_transform(points)
        #points = TimeSeriesScalerMeanVariance().fit_transform(points)

        # take unqiue values before random sampling, so random sample isn't replication of same points
        while len(forest) < self.num_trees:
            # Select random subsets of points uniformly from point set
                ixs = np.random.choice(self.num_points, self.tree_size, replace=False)
            # Add sampled trees to forest
                tree = rrcf_base.RCTree(points[ixs], index_labels=ixs)
                forest.append(tree)
        self.forest = forest

    def batch_anomaly_scores(self):
        '''
            Computes anomaly scores for all points in the rcrf by computing the average
            collusive displacement. Higher scores indicate a higher displacement and thus a 
            higher likelihood of anomaly.

            Returns:
                anomaly_scores: pandas Series with index of points and average collusive 
                                displacement (anomaly score) for each point
        '''
        # assert that fit function has already been called and a forest exists
        try: 
            assert(self.forest is not None)
        except:
            raise ValueError("Cannot compute anomaly scores on forest that has not been fit")

        n = self.num_points
        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)
        for tree in self.forest:
            codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index
        return avg_codisp
    

    def anomaly_score(self, points):
        '''
        Computes anomaly score for point by inserting point in each tree in the rcrf and
        computing the average collusive displacement. Higher scores indicate a higher displacement 
        and thus a higher likelihood of anomaly. It then deletes points from each tree to 
        maintain the state of the forest. (If you want new points to be added to trees, you can 
        use the function stream_anomaly_scores with `new_forest` == False)

        Parameters:
            points:   the points on which to calculate anomaly scores. np.ndarray of size (n x d)

        Returns:
            anomaly_score: pandas Series with index of points and average collusive 
                           displacement (anomaly score) for each point
        '''
         # assert that fit function has already been called and a forest exists
        try: 
            assert(self.forest is not None)
        except:
            raise ValueError("Cannot compute anomaly scores on forest that has not been fit")

        # assert that points.shape has two dimensions
        try:
            assert(len(points.shape) is 2)
        except:
            raise ValueError("Points must have shape (n x d)")

        # assert that dimension of points is the same as dimension in fit tree
        try:
            assert(points.shape[1] == self.dimension)
        except:
            raise ValueError("Forest and points must have the same dimension. Points dim: {} vs. Forest dim: {}".format(points.shape[1], self.dimension))

        # scale point to between 0 and 1 ??? might improve
        #points = TimeSeriesScalerMinMax().fit_transform(points)
        #point = TimeSeriesScalerMeanVariance().fit_transform(pointt)

        avg_codisp = pd.Series(0.0, index=np.arange(points.shape[0]))
        points = rrcf_base.shingle(points, size = 1)
        for index, point in enumerate(points):
            for tree in self.forest:
                tree.insert_point(point, index='point')
                codisp = tree.codisp('point')
                avg_codisp[index] += codisp
                tree.forget_point('point')
        return avg_codisp // self.num_trees

    def stream_anomaly_scores(self, points, window_size, new_forest = False):
        '''
        Computes anomaly scores for all points in a stream by computing the average
        collusive displacement. The assumption is that each point in the stream is only observed
        sequentially. Higher scores indicate a higher displacement and thus a 
        higher likelihood of anomaly. If existing forest does not exist, or existing forest does
        exist with a different window size, create a new forest starting with the first point 
        in the stream. 

        Parameters:
            points:         the stream of point on which to calculate anomaly scores
            window_size:    the window size in which to ingest points. points are mapped as a 
                            n-dimensional window, where n = window_size
            new_forest:     boolean that identifies whether to create a new forest or not

        Returns:
            anomaly_scores: pandas Series with index of points and average collusive 
                            displacement (anomaly score) for each point
        '''

        # create a new empty forest if forest does not exit or forest does exist, but 
        # with different window size
        if self.forest is None or self.dimension is not window_size or new_forest:
            self.num_points = 0
            forest = []
            for _ in range(self.num_trees):
                tree = rrcf_base.RCTree()
                forest.append(tree)
            self.forest = forest
        
        # create rolling window of size window_size
        points_gen = rrcf_base.shingle(points, size=window_size)

        # calculate streaming anomaly scores
        avg_codisp = pd.Series(0.0, index=np.arange(self.num_points, self.num_points + points.shape[0]))
        initial_index = self.num_points
        for index, point in enumerate(points_gen):
            index += initial_index
            for tree in self.forest:
                # If tree is above permitted size, drop the oldest point (FIFO)
                if len(tree.leaves) >= self.tree_size:
                    tree.forget_point(index - self.tree_size)
                    self.num_points -= 1
                # Insert the new point into the tree
                try:
                    tree.insert_point(point, index=index)
                except:
                    ValueError('failure for point {} at index {}'.format(point, index))
                self.num_points += 1
                # Compute codisp on the new point and take the average among all trees
                avg_codisp[index] += tree.codisp(index)
        return avg_codisp // self.num_trees

# main method for testing class
if __name__ == '__main__':

    ''' Preprocessing '''
    # load Twitter hashtag data
    npzfile = np.load("/Users/jeffreygleason 1/Desktop/NewKnowledge/Code/time-series/tapir/tapir/data/prep/hashtags_2019_01_01_to_2019_02_01.npz")
    hashtags = npzfile['rate_vals']
    ht = 0

    # mark top 5% of rate values as anomalous
    anom_thresh = 95
    #anom = hashtags[ht] > np.percentile(hashtags[ht], anom_thresh)

    ''' Instantiate RRCF'''
    # set HPs and instantiate
    num_trees = 200
    tree_size = 40
    clf = robust_rcf(num_trees, tree_size)
    
    ''' Test Different Methods'''
    # test batch anomaly detection
    clf.fit_batch(hashtags[ht].reshape(-1,1))
    anom_score = clf.batch_anomaly_scores()

    # test point anomaly detection 
    # slightly different anomaly scores than batch because inserting every point into every tree 
    # (vs random sampling in batch)
    point_scores = clf.anomaly_score(hashtags[ht].reshape(-1,1))
    
    # mark top 5% of predictions as anomalous
    anom_pred = anom_score > np.percentile(anom_score, anom_thresh)

    # print evaluation
    # print(evaluate(anom, anom_pred)) 

    # plot comparison of labeled anomalies to predicted anomalies
    colors = ('blue', 'red')
    targets = ('non-anomalous', 'anomalous')
    #indices = (np.where(~anom), np.where(anom))
    #data = (hashtags[ht][~anom], hashtags[ht][anom])
    pred_indices = (np.where(~anom_pred), np.where(anom_pred))
    pred_data = (hashtags[ht][~anom_pred], hashtags[ht][anom_pred])
    #plt.subplot(2,1,1)
    #for index, dat, color, target in zip(indices, data, colors, targets):
    #    plt.scatter(index, dat, c = color, label = target, s=10)
    #plt.legend()
    plt.subplot(2,1,1)
    for index, dat, color, target in zip(pred_indices, pred_data, colors, targets):
        plt.scatter(index, dat, c = color, label = target, s=10)
    plt.legend()
    #plt.show()
    
    # test streaming anomaly detection
    window_size = 4
    anom_score = clf.stream_anomaly_scores(hashtags[ht], window_size, new_forest = True)
    # mark top 5% of predictions as anomalous
    anom_pred = anom_score > np.percentile(anom_score, anom_thresh)

    # print evaluation
    #print(evaluate(anom, anom_pred)) 

    # plot comparison of labeled anomalies to predicted anomalies
    pred_indices = (np.where(~anom_pred), np.where(anom_pred))
    pred_data = (hashtags[ht][~anom_pred], hashtags[ht][anom_pred])
    plt.subplot(2,1,2)
    for index, dat, color, target in zip(pred_indices, pred_data, colors, targets):
        plt.scatter(index, dat, c = color, label = target, s=10)
    plt.legend()
    plt.show()

    ## TODO: Test streaming anomaly detection from created tree
    #print(clf.stream_anomaly_scores(hashtags[ht], window_size))






