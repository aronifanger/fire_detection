# -*- coding: utf-8 -*-
from flask import Flask
from classifier import predict_fire

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/user/<username>')
def user(username):
    return username

@app.route('/classifier/<name>')
def classifier(name):
    path = '/home/notebooks/rnpi-all-iamaron-master/src/datasets/Validation/fire/'
    return predict_fire(path+name)

if __name__ == "__main__":
    app.run(host='172.17.0.2',port='5010')