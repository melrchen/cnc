from __future__ import print_function
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
    
# CONV LAYER MODEL AT THE END
batch_size = 5
epochs = 20
image_input = Input(shape=(224, 224, 208))

print('Loading data...')
# Load in data
(x_train, y_train), (x_test, y_test) = pp.load_data()
y_train = np.reshape(y_train, tuple(list(y_train.shape) + [1]))
y_test = np.reshape(y_test, tuple(list(y_test.shape) + [1]))

print('Loaded data')

# MINI MODEL WITH U, V CHANNELS
model = Sequential()
model.add(Conv2D(128, (3,3), strides = (1,1), padding = 'same', activation="relu",
    input_shape = (224, 224, 208)))
model.add(BN())
model.add(Conv2D(64, (3,3), strides = (1,1), padding = 'same', activation = "relu"))
model.add(BN())
model.add(Conv2D(1, (3,3), strides = (1,1), padding = 'same', activation = "tanh"))

# Get two branches
U = model(image_input)
V = model(image_input) 

output = keras.layers.concatenate([U, V], axis=1) # Concatenate U, V outputs

# FINAL MODEL TAKING INTO ACCOUNT BRANCHING/LOSS
final_model = Model(inputs = image_input, outputs = output)

final_model.compile(loss=keras.losses.mean_squared_error,
              optimizer='sgd')

print('Compiled Model')

# Neat thing that lets you view your accuracy over time
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

history = AccuracyHistory()

print('======TRAINING======')
print('x shape: ',x_train.shape)
print('y shape: ', y_train.shape)

final_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

# print("=======VALIDATING=======")
# score = final_model.evaluate(x_test, y_test, verbose=0)
# print('Validation loss:', score[0])
# print('Validation accuracy:', score[1])

# Plot the accuracy vs. epochs
# plt.plot(range(1, epochs + 1), history.val_loss)
# plt.xlabel('Epochs')
# plt.ylabel('Validation Loss')
# plt.show()

# plt.plot(range(1, epochs + 1), history.loss)
# plt.xlabel('Epochs')
# plt.ylabel('Training Loss')
# plt.show()

print("======SAVING======")
model_json = model.to_json()

# Serialize model to JSON
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5
model.save_weights("beachmodel.h5")

if __name__ == "__main__":
    # fit an image
    pass