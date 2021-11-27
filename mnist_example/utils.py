from sklearn.model_selection import train_test_split
from sklearn import svm, tree, metrics
from skimage.transform import rescale
import numpy as np
from joblib import dump, load
from os import path as osp
import os


def preprocess(data, rescale_factor):
    resized_images = []
    for d in data.images:
        resized_images.append(rescale(d, rescale_factor, anti_aliasing=False))
    # flatten the images
    resized_images = np.array(resized_images)
    n_samples = len(resized_images)
    X = resized_images.reshape((n_samples, -1))
    return X, data.target


def create_split(X, y, val_test_ratio=(0.25, 0.5)):
    valid_ratio = val_test_ratio[0]
    test_ratio = val_test_ratio[1]
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(
        X, y, test_size=test_ratio + valid_ratio, shuffle=False
    )

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test_valid,
        y_test_valid,
        test_size=valid_ratio / (test_ratio + valid_ratio),
        shuffle=False,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def str_dict(d):
    out = ""
    for k, v in d.items():
        out += str(k)+"_"
        out += str(v)+"_"
    return out[:-1]

def run_classification_train_task(clf_class, curr_params, X_train, y_train):
    # Initialize model
    clf = clf_class(**curr_params)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    return clf

def test(model, X_train, y_train, X_val, y_val, X_test, y_true):
    
    # Predict the value of the digit on the train subset
    predicted = model.predict(X_train)
    train_acc = metrics.accuracy_score(
        y_true = y_train, y_pred = predicted, normalize=True
        )
    
    # Predict the value of the digit on the validation subset
    predicted = model.predict(X_val)
    val_acc = metrics.accuracy_score(
        y_true = y_val, y_pred = predicted, normalize=True
        )

    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)
    test_acc = metrics.accuracy_score(
        y_true = y_true, y_pred = predicted, normalize=True
        )

    
    return {"train_acc" : train_acc, "val_acc" : val_acc, "test_acc": test_acc}

def load_saved_model(val_test_ratio, rescale_factor, clf_params):
    # load model from disk
    model_folder = osp.abspath("mnist_example/models/tt_{}_val_{}_rescale_{}_hyperparams_{}".format(
            val_test_ratio[1], val_test_ratio[0], rescale_factor, str_dict(clf_params)
        ))
    clf = load(osp.join(model_folder, "model.joblib"))
    return clf

def run_loop_on_hyperparams(clf_class, clf_params, X_train, y_train, X_val, y_val, val_test_ratio, rescale_factor):
    clf = run_classification_train_task(clf_class, curr_params, X_train, y_train, val_test_ratio, rescale_factor)

    return best_clf, max_valid_f1_model, best_hyperparams


def save_best_model(best_clf, clf_class):
    best_model_save_folder = osp.abspath(f"mnist_example/models/best_model_{clf_class.__name__}")
    os.mkdir(best_model_save_folder) if not osp.exists(best_model_save_folder) else None
    dump(best_clf, osp.join(best_model_save_folder, "model.joblib"))
