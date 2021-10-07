"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)


# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
from os import path as osp
# import matplotlib.pyplot as plt
# import pickle
from numpy.lib.npyio import save
from joblib import load
from utils import preprocess, create_split, create_model_train_and_dump, test
# Import datasets, classifiers and performance metrics
from sklearn import datasets
import pandas as pd


###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

out = pd.DataFrame(columns=["Val-Test-Split-Ratios", "Rescale Factor", "F1-Validation", "Test Accuracy", "Optimal Gamma"])
# for val_test_ratio in [(0.15, 0.15), (0.15, 0.30), (0.25, 0.25), (0.3, 0.3), (0.2, 0.4)]:
    # for rescale_factor in [0.25, 0.5, 1, 1.5, 2, 2.5]:
for val_test_ratio in [(0.15, 0.15)]:
    for rescale_factor in [1]:
        # print()
        # print("="*100)
        # print(f"Processing for Val-test ratio {val_test_ratio}, rescale {rescale_factor}")
        # print("="*100)
        # preprocess data
        X, y = preprocess(digits, rescale_factor=rescale_factor)

        # split into train, val and test subsets
        X_train, X_val, X_test, y_train, y_val, y_test = create_split(X, y, val_test_ratio=val_test_ratio)

        #!TODO: Convert into command-line program using argparse
        # Create a classifier: a support vector classifier
        max_f1 = 0
        candidates = []
        for gamma in (10**exp for exp in range(-7,4)):

            clf = create_model_train_and_dump(X_train, y_train, gamma, val_test_ratio, rescale_factor)

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
                "gamma": gamma
            }
            candidates.append(candidate)
            max_f1 = metrcs["f1"] if max_f1 < metrcs["f1"] else max_f1

        # select best candidate model on the basis of f1 score on validation
        max_valid_f1_model = max(candidates, key = lambda x: x["f1_valid"])
        gamma = max_valid_f1_model["gamma"]

        # load model from disk
        best_model_folder = osp.abspath("models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                val_test_ratio[1], val_test_ratio[0], rescale_factor, gamma
            ))
        clf = load(osp.join(best_model_folder, "model.joblib"))

        # print optimal gamma and metrics
        # print(f"Optimal Gamma value: {max_valid_f1_model['gamma']}, on validation subset, \
        #     F1 score: {max_valid_f1_model['f1_valid']}, accuracy: {max_valid_f1_model['accval']}")
        # clf = pickle.loads(saved_model)

        # infer on test dataset
        results = test(model=clf, X_test=X_test, y_true=y_test)
        print(f"Val-test ratio {val_test_ratio}, rescale {rescale_factor}: Test acc: {results['acc']}")
        other = pd.DataFrame({
            "Val-Test-Split-Ratios": f"{val_test_ratio[0]}, {val_test_ratio[1]}",
            "Rescale Factor": rescale_factor,
            "F1-Validation": max_valid_f1_model['f1_valid'], 
            "Test Accuracy": results['acc'], 
            "Optimal Gamma": gamma
            }, index=[0])
        out = out.append(other, ignore_index=True)
print(out.to_markdown())
