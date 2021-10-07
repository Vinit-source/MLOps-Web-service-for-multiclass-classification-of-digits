
from mnist_example import utils
from sklearn import datasets
import os

val_test_ratio = (0.3, 0.3)
rescale_factor = 1
gamma = 0.001

def test_load_data_with_splits():
    digits = datasets.load_digits()
    X, y = utils.preprocess(digits, rescale_factor=rescale_factor)
    X_train, X_val, X_test, y_train, y_val, y_test = utils.create_split(X, y, val_test_ratio=val_test_ratio)
    return X_train, X_val, X_test, y_train, y_val, y_test

def test_create_model_train_and_dump():
    X_train, _, _, y_train, _, _ = test_load_data_with_splits()
    clf = utils.create_model_train_and_dump(X_train, y_train, gamma, val_test_ratio, rescale_factor)
    assert os.path.isfile("mnist_example/models/tt_{}_val_{}_rescale_{}_gamma_{}/model.joblib".format(
        val_test_ratio[1], val_test_ratio[0], rescale_factor, gamma
    ))

def test_small_data_overfit_checking():
    X_train, _, _, y_train, _, _ = test_load_data_with_splits()
    clf = utils.create_model_train_and_dump(X_train, y_train, gamma, val_test_ratio, rescale_factor)
    train_metrics = utils.test(clf, X_train, y_train)
    acc_threshold, f1_threshold = 0.9, 0.9
    assert(train_metrics['acc']>acc_threshold)
    assert(train_metrics['f1']>f1_threshold)