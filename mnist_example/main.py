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
import os
from numpy.lib.npyio import save
from joblib import load
from utils import *
# Import datasets, classifiers and performance metrics
from sklearn import svm, tree, metrics
from sklearn import datasets
# Utils libraries
import pandas as pd
from matplotlib import pyplot as plt

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

# Paths
CURR_DIR = "."
MODELS_DIR = f"{CURR_DIR}/mnist_example/models"

# Make model directory if not exists
os.makedirs(MODELS_DIR) if not osp.exists(MODELS_DIR) else None
# Create output dataframe
out = pd.DataFrame(columns=["Split", "SVM: Test Accuracy", "Gamma", "Decision Tree: Test Accuracy", "Depth"])
#!TODO: Convert into command-line program using argparse

# Declare classifier params
svm_params = {"gamma": [10**i for i in range(-4, 1)]}
tree_params = {"max_depth": [i for i in range(1,11, 2)]}
train_range = range(10, 110, 10)
results_svm_overall, results_tree_overall = [], []
# Main loop
for loop in range(5):
    for val_test_ratio in [(0.1, 0.1)]:
        for rescale_factor in [1]:
            # preprocess data
            X, y = preprocess(digits, rescale_factor=rescale_factor)

            # split into train, val and test subsets
            X_train, X_val, X_test, y_train, y_val, y_test = create_split(X, y, val_test_ratio=val_test_ratio)
            
            results_svm, results_tree = [], []
            for size in train_range:
                # sample out ratio amount of rows from train set
                idx = np.random.choice(np.arange(X_train.shape[0]), int(size*0.01*X_train.shape[0]), replace=False)
                X_train_sampled, y_train_sampled = X_train[idx, :], y_train[idx]

                # Create a support vector classifier
                clf_class = svm.SVC
                best_clf, max_valid_f1_model, best_hyperparams_svm = run_loop_on_hyperparams(clf_class, svm_params, X_train_sampled, y_train_sampled, X_val, y_val, val_test_ratio, rescale_factor)

                # infer on test dataset
                results_svm.append( test(model=best_clf, X_test=X_test, y_true=y_test) )

                # save best SVM model
                save_best_model(best_clf, clf_class)
                
                # Create a Decision tree classifier
                clf_class = tree.DecisionTreeClassifier
                best_clf, max_valid_f1_model, best_hyperparams_tree = run_loop_on_hyperparams(clf_class, tree_params, X_train_sampled, y_train_sampled, X_val, y_val, val_test_ratio, rescale_factor)

                # save best Decision Tree model
                save_best_model(best_clf, clf_class)

                # infer on test dataset
                results_tree.append( test(model=best_clf, X_test=X_test, y_true=y_test) )
    
    results_svm_overall.append(results_svm)
    results_tree_overall.append(results_tree)

# Plot F1 score analysis line chart
compare_f1(results_svm_overall, results_tree_overall, train_range)

# Compare confusion matrices
compare_cms(train_range, results_svm, "SVM")
compare_cms(train_range, results_tree, "Decision Tree")

# Display plots
plt.show()