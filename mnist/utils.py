
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale
import numpy as np

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
    # Split data into train and test dataset 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_ratio, shuffle=False)
    # Split train data into train and validation subsets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size = valid_ratio, shuffle=False)
    return (X_train, X_val, X_test, y_train, y_val, y_test)

def create_model_and_train(X_train, y_train, gamma):
    # create model
    clf = svm.SVC(gamma=gamma)
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
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
