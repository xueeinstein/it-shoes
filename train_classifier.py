'''
Train classifier
--------------------
Train classifier from feature csv files
'''
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

import config

POS, NEG = (1, 0)


def generate_tr_te_data(data_type, ratio=0.75):
    """generate training and test data"""
    if data_type == POS:
        csv_file = config.pos_features_csv
    elif data_type == NEG:
        csv_file = config.neg_features_csv

    data_frame = pd.read_csv(csv_file)
    feature_mat = data_frame.values[:, 1:]
    np.random.shuffle(feature_mat)
    split = int(feature_mat.shape[0] * ratio)

    tr_mat = feature_mat[:split]
    te_mat = feature_mat[split:]

    if data_type == POS:
        tr_labels = np.ones((split,))
        te_labels = np.ones((feature_mat.shape[0] - split))
    elif data_type == NEG:
        tr_labels = np.zeros((split,))
        te_labels = np.zeros((feature_mat.shape[0] - split))

    return tr_mat, tr_labels, te_mat, te_labels


def train_svm_classifier():
    """train SVM classifier"""
    print 'Start to load data'
    ptr_mat, ptr_labels, pte_mat, pte_labels = generate_tr_te_data(POS)
    ntr_mat, ntr_lables, nte_mat, nte_labels = generate_tr_te_data(NEG)

    tr_mat = np.concatenate((ptr_mat, ntr_mat), axis=0)
    tr_labels = np.concatenate((ptr_labels, ntr_lables), axis=0)
    te_mat = np.concatenate((pte_mat, nte_mat), axis=0)
    te_labels = np.concatenate((pte_labels, nte_labels), axis=0)

    print 'Data loaded'
    print 'Training a linear SVM model'
    classifier = LinearSVC()
    classifier.fit(tr_mat, tr_labels)

    print 'Testing linear SVM model'
    pred = classifier.predict(te_mat)
    rights = 0.0
    for i in xrange(pred.shape[0]):
        if pred[i] == te_labels[i]:
            rights += 1

    print 'Accuracy:', rights / pred.shape[0]

    joblib.dump(classifier, config.model_path)
    print 'Classifier saved to', config.model_path


if __name__ == '__main__':
    train_svm_classifier()
