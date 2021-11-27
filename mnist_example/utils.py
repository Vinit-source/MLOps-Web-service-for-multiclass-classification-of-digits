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

def run_classification_train_task(clf_class, curr_params, X_train, y_train, val_test_ratio, rescale_factor):
    # Initialize model
    clf = clf_class(**curr_params)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # store model to disk
    output_folder = osp.abspath("mnist_example/models/tt_{}_val_{}_rescale_{}_hyperparams_{}".format(
        val_test_ratio[1], val_test_ratio[0], rescale_factor, str_dict(curr_params)
    ))
    os.mkdir(output_folder) if not osp.exists(output_folder) else None
    dump(clf, osp.join(output_folder, "model.joblib"))

    return clf

def test(model, X_test, y_true):
    # Predict the value of the digit on the validation subset
    predicted = model.predict(X_test)
    acc = metrics.accuracy_score(
        y_true = y_true, y_pred = predicted, normalize=True
        )
    f1 = metrics.f1_score(
        y_true=y_true, y_pred = predicted, average="macro"
        )

    return {"acc" : acc, "f1" : f1}

def load_saved_model(val_test_ratio, rescale_factor, max_valid_f1_model, clf_params):
    # Extract best hyperparams
    best_hyperparams = dict((k, max_valid_f1_model[k]) for k in clf_params.keys())

    # load model from disk
    best_model_folder = osp.abspath("mnist_example/models/tt_{}_val_{}_rescale_{}_hyperparams_{}".format(
            val_test_ratio[1], val_test_ratio[0], rescale_factor, str_dict(best_hyperparams)
        ))
    clf = load(osp.join(best_model_folder, "model.joblib"))
    return clf, best_hyperparams

def run_loop_on_hyperparams(clf_class, clf_params, X_train, y_train, X_val, y_val, val_test_ratio, rescale_factor):
    max_f1 = 0
    candidates = []
    if len(clf_params) > 0:
        for hyperparam, range_of_vals in clf_params.items():
            for value in range_of_vals:
                curr_params = {hyperparam: value}
                clf = run_classification_train_task(clf_class, curr_params, X_train, y_train, val_test_ratio, rescale_factor)
                
                # Predict the value of the digit on the validation subset
                metrcs = test(clf, X_val, y_val)
                
                # skip irrelevant models on the basis of f1 score
                if max_f1 >= metrcs["f1"]:
                    # print(f"Skipping for gamma = {gamma}")
                    continue

                # save values of relevant models on the basis of f1 score
                candidate = {
                    "accval":metrcs["acc"],
                    "f1_valid": metrcs["f1"],
                }
                candidate.update(curr_params)
                candidates.append(candidate)
                max_f1 = metrcs["f1"] if max_f1 < metrcs["f1"] else max_f1
        
    else:
        clf = run_classification_train_task(clf_class, clf_params, X_train, y_train, val_test_ratio, rescale_factor)
        # Predict the value of the digit on the validation subset
        metrcs = test(clf, X_val, y_val)

        # save values of relevant models on the basis of f1 score
        candidate = {
            "accval":metrcs["acc"],
            "f1_valid": metrcs["f1"],
        }
        candidates.append(candidate)
        max_f1 = metrcs["f1"] if max_f1 < metrcs["f1"] else max_f1

    # select best candidate model on the basis of f1 score on validation
    max_valid_f1_model = max(candidates, key = lambda x: x["f1_valid"])
    best_clf, best_hyperparams = load_saved_model(val_test_ratio, rescale_factor, max_valid_f1_model, clf_params)
    best_model_save_folder = osp.abspath("mnist_example/models/best_model")
    os.mkdir(best_model_save_folder) if not osp.exists(best_model_save_folder) else None
    dump(clf, osp.join(best_model_save_folder, "model.joblib"))
    return best_clf, max_valid_f1_model, best_hyperparams

