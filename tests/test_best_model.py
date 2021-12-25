from sklearn.datasets import load_digits
from mnist_example import utils
from joblib import load
from sklearn.metrics import confusion_matrix
import numpy as np

# load images for test
digits = load_digits()
X, y = utils.preprocess(data=digits, rescale_factor=1)

# Load SVM model
best_model_path = "mnist_example/models/best_model_SVC/model.joblib"
clf = load(best_model_path)

def test_SVC_digit_correct_0():
    i = 0
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_1():
    i = 1
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_2():
    i = 2
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_3():
    i = 3
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_4():
    i = 4
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_5():
    i = 5
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_6():
    i = 6
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_7():
    i = 7
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_8():
    i = 8
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_SVC_digit_correct_9():
    i = 9
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

# load Decision Tree best model
best_model_path = "mnist_example/models/best_model_DecisionTreeClassifier/model.joblib"
clf = load(best_model_path)

def test_Tree_digit_correct_0():
    i = 0
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_1():
    i = 1
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_2():
    i = 2
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_3():
    i = 3
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_4():
    i = 4
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_5():
    i = 5
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_6():
    i = 6
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_7():
    i = 7
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_8():
    i = 8
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

def test_Tree_digit_correct_9():
    i = 9
    idx = len(y) - np.argmax((y==i)[::-1]) - 1
    # print(y[idx], idx)
    prediction = clf.predict(X[idx].reshape((1, -1)))
    assert(prediction==y[idx])

# def test_Tree_class_acc_0():
    # use confusion_matrix
    # cm.diagonal
    # Reference: https://stackoverflow.com/a/50977153/15294463