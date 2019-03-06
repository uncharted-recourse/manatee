# New Knowledge's Shapelet Time Series Classifier

Assumptions in data labeling:

1. 419 emails are labeled as spam

2. enron emails are labeled as not spam

3. All JPL data abuse dataset emails are treated as foe - exceptions are FalsePositive and Recon which were dropped (former due to self-explanatory reason, and latter due to "lack of full understanding" reasons). Unknown was dropped as well.

Built on top of the shapelet convolutional neural network time series classification model:

https://github.com/NewKnowledge/sloth/blob/master/Sloth/classify.py

# Switching Binary and Multiclass Classifiers

You can swap out the multiclass multilabel model for the binary model (enabled by default) by modifying `config.ini` as specified in `deployed_checkpoints/checkpoint_descriptions.txt`

Be sure to rebuild docker images, if using dockerized version of code, after making this edit.

# gRPC Dockerized Classifier

The gRPC interface consists of the following components:
*) `grapevine.proto` in `protos/` which generates `grapevine_pb2.py` and `grapevine_pb2_grpc.py` according to instructions in `protos/README.md` -- these have to be generated every time `grapevine.proto` is changed
*) `shapelet_clf_server.py` which is the main gRPC server, serving on port `50052` (configurable via `config.ini`)
*) `shapelet_clf_client.py` which is an example script demonstrating how the main gRPC server can be accessed to classify emails 
 
To build corresponding docker image:
`sudo docker build -t nk-shapelet-classifier:latest .`

To run docker image, simply do
`sudo docker run -it -p 50052:50052 nk-shapelet-classifier:latest`

Finally, run the test script as `python3 shapelet_clf_client.py`

