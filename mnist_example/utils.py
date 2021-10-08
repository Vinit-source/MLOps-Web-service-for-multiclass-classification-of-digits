
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
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

def create_model_train_and_dump(X_train, y_train, gamma, val_test_ratio, rescale_factor):
    # create model
    clf = svm.SVC(gamma=gamma)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # store model to disk
    output_folder = osp.abspath("mnist_example/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
        val_test_ratio[1], val_test_ratio[0], rescale_factor, gamma
    ))
    os.mkdir(output_folder) if not osp.exists(output_folder) else None
    # with open(osp.join(output_folder, "model.pkl"), 'wb') as fp:
    #     pickle.dump(candidate, fp)
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
