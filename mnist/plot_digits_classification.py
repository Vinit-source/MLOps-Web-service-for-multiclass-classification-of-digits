"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import data
from skimage.transform import resize
import numpy as np


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
print(f"Original Image size: {digits.images[0].shape}")
for size in [16, 32, 64]:

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
    digits.images = np.array([resize(image, (size, size)) for image in digits.images])
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    for split in [0.1, 0.2, 0.3, 0.4]:
        X_train, X_test, y_train, y_test = train_test_split(
            data, digits.target, test_size=split, shuffle=False)

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)

        ###############################################################################
        # Below we visualize the first 4 test samples and show their predicted
        # digit value in the title.

    ###############################################################################
        acc = metrics.accuracy_score(y_test, predicted, normalize=True)
        f1 = metrics.f1_score(y_test, predicted, average="micro")
        print(f"{size}x{size}\t\t{split}\t\t{acc*100:.2f}\t\t{f1:.2f}")

        ###############################################################################
    print()
