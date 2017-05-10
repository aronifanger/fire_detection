# -*- coding: utf-8 -*-
from flask import Flask
from flask import request
from classifier import predict_from_path
from classifier import predict_from_url
from classifier import predict_from_image

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello!'

@app.route('/predict')
def predict():
    
    url = request.args.get('url')
    #path = request.args.get('path')
    
    if(url):
        return predict_from_url(url)
    #elif(path):
    #    return predict_from_path(path)
    else:
        return 'Caminho não encontrado.'
    
@app.route('/predict/upload', methods=['POST'])
def upload():
    try:
        imagefile = flask.request.files.get('imagefile', '')
        return imagefile.__class__.__name__
    except Exception as err:
        return err

if __name__ == "__main__":
    app.run(host='172.17.0.2',port='5010')