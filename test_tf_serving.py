
# import modules
import argparse
import json

# import the necessary packages
import numpy as np
import requests
from keras.applications import inception_v3
from keras.preprocessing import image

# set up the argument parser and parse the args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

# grab the image path
image_path = args['image']

# preprocess the image to fit inception v3 requirement
img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255. # divide by 255. because we need our channels in [0, 1]
# define payload
payload = {
    "instances": [{'input_image': img.tolist()}]
}

# send the request to tf serving
r = requests.post('http://localhost:8501/v1/models/inception:predict', json=payload)

# decode response
pred = json.loads(r.content.decode('utf-8'))

# print the prediction classes
print(len(pred['predictions'][0]))

# decode_prediction do the job for us and find the top 5 and their %
print(json.dumps(inception_v3.decode_predictions(np.array(pred['predictions']))[0]))