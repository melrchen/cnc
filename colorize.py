import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model, Sequential, model_from_json
from keras.layers import Conv2D, Input, BatchNormalization as BN
# import matplotlib.pyplot as plt
import color_preprocessing as pp
import cv2
import scipy as sp
import numpy as np
import classify as cl 
import os

# Load classifier
# open model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer=keras.optimizers.Adam(),
          metrics=['accuracy'])

# Load colorizers
colorizers = []
scenes = ['plain', 'beach', 'mountain', 'cave', 'city', 'general']
for scene in scenes:
    # open model
    json_file = open('{}model.json'.format(scene), 'r')
    loaded = json_file.read()
    json_file.close()
    loaded = model_from_json(loaded)

    # load weights into new model
    loaded.load_weights("{}model.h5".format(scene))
    colorizers.append(loaded)



def classify(filepath):
    '''
    Generates a probability vector (5 x 1)
    on classes for the filepath, and chooses class from the 
    filepath.
    '''
    prob_vector = cl.predict_class(filepath, loaded_model)
    # print(prob_vector)
    
    scene = np.random.choice(5, 1, p=prob_vector[0]) # choose
    
    return scene


def color(filepath):
    '''
    Returns the concatenated result of the convnet based on scene result.
    '''
    scene = classify(filepath)
    print('Scene: ', scenes[scene[0]])

    # load model
    json_file = open('plainmodel.json'.format(scene), 'r')
    loaded_model = model_from_json(json_file.read())
    json_file.close()

    # load model weights and colorize
    loaded_model.load_weights('{}model.h5'.format(scenes[scene[0]]))
    loaded_model.compile(loss=keras.losses.mean_squared_error,
              optimizer='sgd')

    inp = np.reshape(pp.upsample_hypercolumn(filepath), (1, 224, 224, 104))
    
    return loaded_model.predict(inp)


def extract_UV(filepath):
    '''
    Returns the U and V channels for the filepath
    '''
    UV = color(filepath)
    



    


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'plaingray.jpeg')

    extract_UV(path)