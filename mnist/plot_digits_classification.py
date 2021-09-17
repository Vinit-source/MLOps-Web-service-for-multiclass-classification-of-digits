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
# from os import access
# import matplotlib.pyplot as plt
import pickle
from numpy.lib.npyio import save

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

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

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# print(f"Gamma\t\tAccuracy\tF1-score (micro)")
# Split data into 70 % train and 30 % held-out 
X_train, X_rem, y_train, y_rem = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False)
# Split held-out data into 50% train and 50% test subsets
X_val, X_test, y_val, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, shuffle=False)

# Create a classifier: a support vector classifier
max_f1 = 0
candidates = []
for gamma in (10**exp for exp in range(-7,4)):
    clf = svm.SVC(gamma=gamma)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the validation subset
    # predicted_train = clf.predict(X_train)
    predicted_val = clf.predict(X_val)

    # acctrain = metrics.accuracy_score(y_train, predicted_train, normalize=True)
    accval = metrics.accuracy_score(y_val, predicted_val, normalize=True)
    f1val = metrics.f1_score(
        y_true=y_val, y_pred = predicted_val, average="macro"
        )
    # print(f"Gamma = {gamma}, Train : Val = {acctrain} : {accval}")

    if max_f1 > f1val:
        print(f"Skipping for gamma = {gamma}")
        continue
    candidate = {
        "model":clf,
        "accval":accval,
        "f1_valid": f1val,
        "gamma": gamma
    }
    candidates.append(candidate)
    max_f1 = f1val if max_f1 < f1val else max_f1
        # saved_model = pickle.dumps(clf)

max_valid_f1_model = max(candidates, key = lambda x: x["f1_valid"])
print(f"Optimal Gamma value: {max_valid_f1_model['gamma']}, on validation subset, \
    F1 score: {max_valid_f1_model['f1_valid']}, accuracy: {max_valid_f1_model['accval']}")
# clf = pickle.loads(saved_model)
clf = max_valid_f1_model["model"]
predicted_test = clf.predict(X_test)
acctest = metrics.accuracy_score(y_test, predicted_test, normalize=True)
print(f"Test accuracy: {acctest}")