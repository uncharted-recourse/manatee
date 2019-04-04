# New Knowledge's Time Series Classification Methods

The acronym manatee stands for **M**ethods for **A**nomaly **N**otification **A**gainst **T**im**E**-series **E**vidence

This directory presents two methods for time series classification:

## 1. Shapelet Classifier

This classifier learns a dictionary of "shapelets," or discriminative subsequences, from the training data (the solutions are not unique). The closest distance between each shapelet and a time series defines a new feature representation, known as the shapelet-transformation. 

The model comes from this paper: https://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf (Grabocka et al. 2014) and is based off of this open-source library: https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.shapelets.html. New Knowledge's implementation of the shapelet classifier, which draws heavily from the previous library is available here: https://github.com/NewKnowledge/sloth/blob/master/Sloth/classify.py.

## 2. FCN-LSTM Classifier

This classifier connects an attention-based LSTM layer to multiple convolution and batch normalization layer sequences. The model originally comes from this paper: https://arxiv.org/abs/1801.04503 (Karim et al. 2018) and is implemented in this github repository: https://github.com/titu1994/LSTM-FCN. New Knowledge's slighty edited implementation is avaiable here: https://github.com/NewKnowledge/LSTM-FCN.

## Training Data for Spam Classification Problem

The data used for the spam classification problem consisted of the 419 Nigerian prince dataset (spam), the Enron dataset (ham), and a private dataset of attack emails from the NASA Jet Propulsion Laboratory (JPL). 

## gRPC Dockerized Classifiers for Deployment

The folders **NK_Shapelet_Classifier** and **NK_FCN_LSTM_Classifier** each contain gRPC interfaces with the following components:

1) `grapevine_pb2.py` and `grapevine_pb2_grpc.py` generated from `grapevine.proto` in `protos/` according to instructions in `protos/README.md`. These files must be generated every time `grapevine.proto` is changed
2) `<classifier_name>_server.py` which is the main gRPC server, serving on port `50050` (configurable via `config.ini`)
3) `<classifier_name>_client.py` which is an example script demonstrating how the main gRPC server can be accessed to classify individual emails (the methods each build streaming rate functions from sequences of email timestamps)
 
To build the corresponding docker image:
`sudo docker build -t <image_name>:latest .`

To run the docker image, simply do
`sudo docker run -it -p 50050:50050 <image_name>:latest`

Finally, you can run the test client script as `python3 <classifier_name>_client.py` (contained in each of the individual folders).

