# import modules
import base64
import json
import urllib
from io import BytesIO

# import the necessary packages
import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.applications import inception_v3
from keras.preprocessing import image

# start flask
app = Flask(__name__)

@app.route('/inception/predict/', methods=['POST'])
def image_classifier():
    print('in the function')
    # Decoding and pre-processing base64 image
    print(request.get_json())
    image_temp = urllib.request.urlretrieve(request.get_json()["image_url"])[0]
    img = image.img_to_array(image.load_img(image_temp, target_size=(224, 224))) / 255.

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:8501/v1/models/inception:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])

app.run()