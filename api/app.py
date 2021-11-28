from flask import Flask, request
import numpy as np
from joblib import load

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/svm_predict", methods=["POST"])
def svm_predict():
    best_model_path = "mnist_example/models/best_model_SVC/model.joblib"
    clf = load(best_model_path)
    print("Model loaded.")
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return str(predicted[0])


@app.route("/tree_predict", methods=["POST"])
def tree_predict():
    best_model_path = "mnist_example/models/best_model_DecisionTreeClassifier/model.joblib"
    clf = load(best_model_path)
    print("Model loaded.")
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return "Prediction" + str(predicted[0]) + "\n"

if __name__ == '__main__':
    app.run(debug=True)
