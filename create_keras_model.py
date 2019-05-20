# import the necessary packages
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# load the model with keras
inception_model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))

# save the model with keras
inception_model.save('./models/keras_inception.h5')