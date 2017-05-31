'''
Train classifier
--------------------
Train classifier from feature csv files
'''
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

POS, NEG = (1, 0)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    copy from
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def split_tr_te_data(data_type, ratio):
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


def generate_tr_te_data(ratio=0.75):
    """generate training and test data"""
    ptr_mat, ptr_labels, pte_mat, pte_labels = split_tr_te_data(POS, ratio)
    ntr_mat, ntr_lables, nte_mat, nte_labels = split_tr_te_data(NEG, ratio)

    tr_mat = np.concatenate((ptr_mat, ntr_mat), axis=0)
    tr_labels = np.concatenate((ptr_labels, ntr_lables), axis=0)
    te_mat = np.concatenate((pte_mat, nte_mat), axis=0)
    te_labels = np.concatenate((pte_labels, nte_labels), axis=0)

    # shuffle
    tr_idxs = np.arange(tr_mat.shape[0])
    te_idxs = np.arange(te_mat.shape[0])
    np.random.shuffle(tr_idxs)
    np.random.shuffle(te_idxs)
    return (tr_mat[tr_idxs], tr_labels[tr_idxs],
            te_mat[te_idxs], te_labels[te_idxs])


def train_svm_classifier(debug=False, n_jobs=-1):
    """train SVM classifier"""
    print('Start to load data')
    tr_mat, tr_labels, te_mat, te_labels = generate_tr_te_data()
    print('Data loaded')
    print('Training a linear SVM model')
    estimator = LinearSVC()
    k_fold = KFold(tr_mat.shape[0], n_folds=5)
    Cs = np.logspace(-10, 0, 10)
    clf = GridSearchCV(estimator=estimator, cv=k_fold, n_jobs=n_jobs,
                       param_grid=dict(C=Cs), verbose=10)
    clf.fit(tr_mat, tr_labels)

    if debug:
        # debug with learning curve
        title = 'Learning Curves(SVM, linear kernel, $C=%.6f$)'\
                % clf.best_estimator_.C
        estimator = LinearSVC(C=clf.best_estimator_.C)
        plot_learning_curve(estimator, title, tr_mat, tr_labels, cv=k_fold,
                            n_jobs=n_jobs)
        plt.savefig('learning_curve.png')

    print('Testing linear SVM model')
    print('Evaluate on test set')
    clf = LinearSVC(C=clf.best_estimator_.C)
    clf.fit(tr_mat, tr_labels)
    score = clf.score(te_mat, te_labels)
    print('score: {}'.format(score))
    print(clf.get_params())

    joblib.dump(clf, config.model_path)
    print('Classifier saved to {}'.format(config.model_path))


if __name__ == '__main__':
    train_svm_classifier(debug=True, n_jobs=20)
