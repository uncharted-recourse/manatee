from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def evaluate(y_truth, y_pred):
    '''
        Evaluation utility prints out the sklearn classification report
        and accuracy score for the comparison between y_truth and y_pred
    '''
    print('Evaluating on {} predictions'.format(len(y_pred)))
    print("Accuracy = ", accuracy_score(y_truth, y_pred))
    target_names = ['non-anomalous', 'anomalous']  # anomalous = predict: 1
    print(classification_report(y_truth, y_pred, target_names=target_names))