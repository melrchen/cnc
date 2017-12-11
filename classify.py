import keras
from keras.layers import Dense, Flatten, Dropout
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json
import classif_preprocessing as pp
import numpy as np 
import os

img_x, img_y = 40, 40

def predict_class(filepath, model):
    '''
    Returns the probability vector for the class of the filepath.
    *Single image*
    '''
    # Preprocess and normalize x (input image)
    x = pp.keras_preprocess(filepath)
    x = x.astype('float32')
    x /= 255
    x = x.reshape(1, img_x, img_y, 1) # reshape into 4D tensor for the convnet

    return model.predict(x)


def extract_hypercolumn(model, layer_indexes, instance):
    test_image = instance
    outputs    = [layer.output for layer in model.layers]          # all layer outputs
    comp_graph = [K.function([model.input]+ [K.learning_phase()], [output]) for output in outputs]  # evaluation functions

    layer_outputs_list = [func(test_image) for func in comp_graph]

    feature_maps = []
    for layerIdx in layer_indexes:
        feature_maps.append(layer_outputs_list[layerIdx][0][0])


    hypercolumns = []
    for idx, convmap in enumerate(feature_maps):
        #        vv = np.asarray(convmap)
        #        print(vv.shape)
        vv = np.asarray(convmap)
        print('shape of feature map at layer ', layer_indexes[idx], ' is: ', vv.shape)

        for i in range(vv.shape[-1]):
            fmap = vv[:,:,i]
            upscaled = sp.misc.imresize(fmap, size=(img_width, img_height),
                                    mode="F", interp='bilinear')
            hypercolumns.append(upscaled)  

    # hypc = np.asarray(hypercolumns)
    # print('shape of hypercolumns ', hypc.shape)

    return np.asarray(hypercolumns)


if __name__ == "__main__":
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

    # make prediction
    path = os.path.join(os.getcwd(), 'plaingray', 'n08659446_1126gray.jpeg')
    # print(predict_class(path, loaded_model))

    # get hypercolumn?
    instance = pp.keras_preprocess(path)
    print(extract_hypercolumn(loaded_model, [0, 2], instance))