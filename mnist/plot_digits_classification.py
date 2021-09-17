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
from os import access
import matplotlib.pyplot as plt

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
# Create a classifier: a support vector classifier
max_acc = 0
for gamma in (10**exp for exp in range(-7,4)):
    clf = svm.SVC(gamma=gamma)

    # Split data into 50% train and 50% test subsets
    X_train, X_rem, y_train, y_rem = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, shuffle=False)
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted_train = clf.predict(X_train)
    predicted_val = clf.predict(X_val)
    predicted_test = clf.predict(X_test)
    
    acctrain = metrics.accuracy_score(y_train, predicted_train, normalize=True)
    accval = metrics.accuracy_score(y_val, predicted_val, normalize=True)
    acctest = metrics.accuracy_score(y_test, predicted_test, normalize=True)
    print(f"Gamma = {gamma}, Train : Val : Test = {acctrain} : {accval} : {acctest}")

    if max_acc < accval:
        max_acc = accval
        optimal_gamma = gamma
        optimal_accs = (acctrain, accval, acctest)

print(f"Optimal Gamma value: {optimal_gamma:.2f}, Train : Val : Test = {optimal_accs[0]} : {optimal_accs[1]} : {optimal_accs[2]}")