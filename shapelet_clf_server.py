#
# GRPC Server for NK Shapelet Classifier
# 
# Uses GRPC service config in protos/grapevine.proto
# 

from flask import Flask, request

import time
import pandas as pd
import numpy as np
import configparser
import tensorflow as tf

from Sloth.classify import Shapelets
from Sloth.preprocess import events_to_rates
from tslearn.preprocessing import TimeSeriesScalerMinMax

import grpc
import logging
import grapevine_pb2
import grapevine_pb2_grpc
from concurrent import futures
import matplotlib.pyplot as plt

# GLOBALS
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEBUG = True # boolean to specify whether or not print DEBUG information

#-----
class NKShapeletClassifier(grapevine_pb2_grpc.ClassifierServicer):

    def __init__(self):
        self.series = []

        # set shapelet HPs
        self.EPOCHS = 100
        self.LENGTH = 0.1
        self.NUM_SHAPELET_LENGTHS = 2
        self.NUM_SHAPELETS = 0.2
        self.LEARNING_RATE = .01
        self.WEIGHT_REGULARIZER = .01

         # set rate function HPs
        self.SERIES_LENGTH = 3600 # series length in seconds
        self.MIN_POINTS = 10
        self.NUM_BINS = 60
        self.FILTER_BANDWIDTH = 1

        # instantiate shapelet clf and model object using deployed weights
        self.clf = Shapelets(self.EPOCHS, self.LENGTH, self.NUM_SHAPELET_LENGTHS, self.NUM_SHAPELETS, self.LEARNING_RATE, self.WEIGHT_REGULARIZER)
        self.model = self.clf.generate_model(int(self.SERIES_LENGTH / 60), len(CATEGORIES.split(',')))
        print('Load weights...')
        self.model.load_weights("deployed_checkpoints/" + MODEL_OBJECT)
        self.model._make_predict_function()
        print('Weights loaded...')
        self.clf.encode(CATEGORIES.split(','))

    # Main classify function
    def Classify(self, request, context):

        # init classifier result object
        result = grapevine_pb2.Classification(
            domain=DOMAIN_OBJECT,
            prediction='false',
            confidence=0.0,
            model="NK_shapelet_classifer",
            version="0.0.1",
            meta=grapevine_pb2.Meta(),
        )

        # get text from input message
        input_time = request.createdAt

        # exception case
        if input_time is None:
            return result

        # Shapelet NK_shapelet_classifier prediction code
        start_time = time.time()
        
        # add new timestamp to time series
        self.series.append(input_time)

        # delete old timestamps if necessary
        while max(self.series) - min(self.series) > (self.SERIES_LENGTH):
            print('Deleting point {} from series'.format(self.series.index(min(self.series))))
            del self.series[self.series.index(min(self.series))] 

        # check if >= min_points exist in the series
        if len(self.series) < self.MIN_POINTS:
            print('There are not enough points in this series to make a shapelet classification.')
            print('This series has {} points, but at least {} are needed for classification'.format(len(self.series), self.MIN_POINTS))
            return result
        
        # transform series to rate function, scale, and make prediction
        max_time = min(self.series) + self.SERIES_LENGTH
        series_values, _ = events_to_rates(self.series, filter_bandwidth = self.FILTER_BANDWIDTH, max_time = max_time, num_bins = self.NUM_BINS, density = True)
        series_values = series_values.reshape((1, len(series_values), 1))
        series_values = TimeSeriesScalerMinMax().fit_transform(series_values)

        y_probs = self.model.predict(series_values)
        print(y_probs)
        pred, confidence = self.clf.decode(y_probs,P_THRESHOLD)
        print("Classification result is (class / confidence): {} / {}".format(pred, confidence))

        elapsed_time = time.time()-start_time
        print("Total time for classification is : %.2f sec" % elapsed_time)

        if pred and confidence: # empty edge case
            result.prediction = pred[0]
            result.confidence = confidence[0]

        return result

#-----
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ClassifierServicer_to_server(NKShapeletClassifier(), server)
    server.add_insecure_port('[::]:' + GRPC_PORT)
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logging.basicConfig() # purpose?
    config = configparser.ConfigParser()
    config.read('config.ini')
    modelName = config['DEFAULT']['modelName']
    print("using model " + modelName + " ...")
    global MODEL_OBJECT
    MODEL_OBJECT = modelName
    categories = config['DEFAULT']['categories']
    print("using categories " + categories + " ...")
    global CATEGORIES
    CATEGORIES = categories
    p_threshold = config['DEFAULT']['p_threshold']
    print("using p_threshold " + p_threshold + " ...")
    global P_THRESHOLD
    P_THRESHOLD = float(p_threshold)
    domain = config['DEFAULT']['domain']
    print("using domain " + domain + " ...")
    global DOMAIN_OBJECT
    DOMAIN_OBJECT = domain
    port_config = config['DEFAULT']['port_config']
    print("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    
    serve()