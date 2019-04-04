#
# GRPC Server for NK FCN-LSTM Classifier
# 
# Uses GRPC service config in protos/grapevine.proto
# 

from flask import Flask, request

import time
import pandas as pd
import numpy as np
import configparser

from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam
from layer_utils import AttentionLSTM
from sklearn.preprocessing import LabelEncoder

from Sloth.preprocess import events_to_rates

import grpc
import logging
import grapevine_pb2
import grapevine_pb2_grpc
from concurrent import futures

# GLOBALS
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

DEBUG = True # boolean to specify whether or not print DEBUG information

restapp = Flask(__name__)

# function that generates attention lstm-fcn model
def generate_alstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=128):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])
    out = Dense(NB_CLASS, activation='softmax')(x)
    model = Model(ip, out)
    return model

#-----
class NKFCNLSTMClassifier(grapevine_pb2_grpc.ClassifierServicer):

    def __init__(self):
        self.series = []

         # set rate function HPs
        self.SERIES_LENGTH = 240 * 60 # series length in seconds
        self.MIN_POINTS = 5
        self.NUM_BINS = 300
        self.FILTER_BANDWIDTH = 2


        # instantiate FCN-LSTM clf and model object using deployed weights
        self.model = generate_alstmfcn(int(self.NUM_BINS), len(CATEGORIES.split(',')))
        self.model.load_weights("deployed_checkpoints/" + MODEL_OBJECT)
        print("Weights loaded from deployed_checkpoints/" + MODEL_OBJECT)
        self.le = LabelEncoder().fit(CATEGORIES.split(','))

    # Main classify function
    def Classify(self, request, context):

        # init classifier result object
        result = grapevine_pb2.Classification(
            domain=DOMAIN_OBJECT,
            prediction='false',
            confidence=0.0,
            model="NK_FCN_LSTLM_classifier",
            version="0.0.1",
            meta=grapevine_pb2.Meta(),
        )

        # get text from input message
        input_time = request.createdAt

        # exception case
        if input_time is None:
            return result

        # CNN LSTM classifier prediction code
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
        series_values, _ = events_to_rates(self.series, filter_bandwidth = self.FILTER_BANDWIDTH, max_time = max_time, min_time = min(self.series), num_bins = self.NUM_BINS, density = True)
        series_values = series_values.reshape((1, 1, len(series_values)))

        print(series_values.shape)

        preds = self.model.predict(series_values)
        y_probs = [p / np.sum(preds) for p in preds]
        result.confidence = y_probs[0] if y_probs[0] > P_THRESHOLD else 1 - y_probs[0]
        result.prediction = self.le.inverse_transform(np.argmax(preds))
        print("Classification result is (class / confidence): {} / {}".format(result.prediction, result.confidence))

        elapsed_time = time.time()-start_time
        print("Total time for classification is : %.2f sec" % elapsed_time)

        return result

#-----
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grapevine_pb2_grpc.add_ClassifierServicer_to_server(NKFCNLSTMClassifier(), server)
    server.add_insecure_port('[::]:' + GRPC_PORT)
    server.start()
    restapp.run()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

@restapp.route("/healthcheck")
def health():
    return "HEALTHY"

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