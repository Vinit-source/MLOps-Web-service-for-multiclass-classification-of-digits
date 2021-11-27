
from mnist_example import utils
from sklearn import datasets
import os
from math import floor, ceil

val_test_ratio = (0.1, 0.2)
rescale_factor = 1
gamma = 0.001

def load_data_with_splits():
    digits = datasets.load_digits()
    X, y = utils.preprocess(digits, rescale_factor=rescale_factor)
    X_train, X_val, X_test, y_train, y_val, y_test = utils.create_split(X, y, val_test_ratio=val_test_ratio)
    return X_train, X_val, X_test, y_train, y_val, y_test

def test_load_data_with_splits():
    digits = datasets.load_digits()
    X, y = utils.preprocess(digits, rescale_factor=rescale_factor)
    n = len(X)
    print(n)
    # X_train, X_val, X_test, y_train, y_val, y_test = utils.create_split(X, y, val_test_ratio=val_test_ratio)
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_with_splits()
    lX_train, lX_val, lX_test, ly_train, ly_val, ly_test = len(X_train), len(X_val), len(X_test), len(y_train), len(y_val), len(y_test)
    assert(lX_train in [floor(0.7*n), ceil(0.7*n)])
    assert(lX_test in [floor(0.2*n), ceil(0.2*n)])
    assert(lX_val in [floor(0.1*n), ceil(0.1*n)])
    assert((lX_train + lX_test + lX_val) == n)
    assert(ly_train in [floor(0.7*n), ceil(0.7*n)])
    assert(ly_test in [floor(0.2*n), ceil(0.2*n)])
    assert(ly_val in [floor(0.1*n), ceil(0.1*n)])
    assert((ly_train + ly_test + ly_val) == n)
    assert(lX_train == ly_train)
    assert(lX_test == ly_test)
    assert(lX_val == ly_val)
    assert(X_train.shape[1:] == X_test.shape[1:])
    assert(X_train.shape[1:] == X_val.shape[1:])


def test_create_model_train_and_dump():
    X_train, _, _, y_train, _, _ = load_data_with_splits()
    clf = utils.create_model_train_and_dump(X_train, y_train, gamma, val_test_ratio, rescale_factor)
    assert os.path.isfile("mnist_example/models/tt_{}_val_{}_rescale_{}_gamma_{}/model.joblib".format(
        val_test_ratio[1], val_test_ratio[0], rescale_factor, gamma
    ))

def test_small_data_overfit_checking():
    X_train, _, _, y_train, _, _ = load_data_with_splits()
    # clf = utils.create_model_train_and_dump(X_train, y_train, gamma, val_test_ratio, rescale_factor)
    try:
        clf = utils.load_saved_model(
            val_test_ratio, rescale_factor, gamma
        )
    except:
        raise Exception("Model cannot be loaded. The model file might be corrupted.")
    train_metrics = utils.test(clf, X_train, y_train)
    acc_threshold, f1_threshold = 0.9, 0.9
    assert(train_metrics['acc']>acc_threshold)
    assert(train_metrics['f1']>f1_threshold)