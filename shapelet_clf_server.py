#
# GRPC Server for NK Shapelet Classifier
# 
# Uses GRPC service config in protos/grapevine.proto
# 

import nltk.data
from random import shuffle
from json import JSONEncoder
from flask import Flask, request

import time
import pandas
import pickle
import numpy as np
import configparser
import os.path
import pandas as pd

from Simon import *
from Simon.Encoder import *
from Simon.DataGenerator import *
from Simon.LengthStandardizer import *

import grpc
import logging
import grapevine_pb2
import grapevine_pb2_grpc
from concurrent import futures


# GLOBALS
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEBUG = True # boolean to specify whether or not print DEBUG information

#-----
class NKShapeletClassifier(grapevine_pb2_grpc.ClassifierServicer):

    def __init__(self):
        self.series = []

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
        input_time = request.datetime

        # Exception cases
        if (len(input_time.strip()) == 0) or (input_time is None):
            return result

        # return if < SERIES_LENGTH time has accumulated
        if len(self.series) < SERIES_LENGTH:
            return result

        # Shapelet NK_shapelet_classifier prediction code
        start_time = time.time()
        # set important parameters
        epochs = 100
        length = 0.1
        num_shapelet_lengths = 1
        num_shapelets = 0.1
        learning_rate = .01
        weight_regularizer = .01
        p_threshold = 0.5 # decision boundary

        ## TODO - update nclasses support for multiclass
        nclasses = 2
        checkpoint_dir = "deployed_checkpoints/"
        
        # add new timestamp to time series
        self.series.append(input_time)

        # delete oldest timestamp if necessary
        if len(self.series) > SERIES_LENGTH:
            self.series.pop(0) 

        clf = Shapelets(epochs, length, num_shapelet_lengths, num_shapelets, learning_rate, weight_regularizer)
        model = clf.generate_model(SERIES_LENGTH, nclasses)
        model.load_weights(MODEL_OBJECT)
        y = model.predict_proba(np.array(self.series).reshape(len(self.series), 1, 1))

        ## TODO - add encoder
        shapelet_result = encoder.reverse_label_encode(y,p_threshold)
        print("Classification result is:")
        print(shapelet_result)

        elapsed_time = time.time()-start_time
        print("Total time for classification is : %.2f sec" % elapsed_time)
        
        clf.clear_session()  # critical for enabling repeated calls of function
        
        # TODO - here
        if shapelet_result: # empty edge case
            result.prediction = 
            result.confidence = 

        return result

#-----
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ClassifierServicer_to_server(NKEmailClassifier(), server)
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
    domain = config['DEFAULT']['domain']
    print("using domain " + domain + " ...")
    global DOMAIN_OBJECT
    DOMAIN_OBJECT = domain
    port_config = config['DEFAULT']['port_config']
    print("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    
    serve()