#
# Test GRPC client code for NK Shapelet Classifier
#
#

from __future__ import print_function
import logging

import grpc
import configparser
import grapevine_pb2
import grapevine_pb2_grpc
import time
import json
import numpy as np
import pandas as pd
from manatee.preprocess import parse_weekly_timestamps

def run():

    channel = grpc.insecure_channel('localhost:' + GRPC_PORT)
    stub = grapevine_pb2_grpc.ClassifierStub(channel)

    # 1 email every 10 seconds - hopefully these will be classified as HAM
    n=100
    timestamp = int(time.time())
    for i in range(n):
        timestamp += 10
        testMessage = grapevine_pb2.Message(
            raw="This raw field isn't used by shapelet classifier, only createdAt field",
            text="This text field isn't used by shapelet classifier, only createdAt field",
            language = "This text field isn't used by shapelet classifier, only createdAt field", 
            createdAt = timestamp
        )
        classification = stub.Classify(testMessage)
        print('Classifier returned this classification for email {} of {} HAM example (class / confidence): {} / {}'.format(i + 1, n, classification.prediction, classification.confidence))
    
    # add spike of 100 emails (1 / second) - hopefully these time series will be classified as SPAM
    n=100
    for i in range(n):
        timestamp += 1
        testMessage = grapevine_pb2.Message(
            raw="This raw field isn't used by shapelet classifier, only createdAt field",
            text="This text field isn't used by shapelet classifier, only createdAt field",
            language = "This text field isn't used by shapelet classifier, only createdAt field", 
            createdAt = timestamp 
        )
        classification = stub.Classify(testMessage)
        print('Classifier returned this classification for email {} of {} SPAM example (class / confidence): {} / {}'.format(i + 1, n, classification.prediction, classification.confidence))

if __name__ == '__main__':
    logging.basicConfig() # purpose?
    config = configparser.ConfigParser()
    config.read('config.ini')
    port_config = config['DEFAULT']['port_config']
    print("using port " + port_config + " ...")
    global GRPC_PORT
    GRPC_PORT = port_config
    run()