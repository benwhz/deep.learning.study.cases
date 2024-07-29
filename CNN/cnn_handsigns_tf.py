import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras
from keras import layers

from cnn_utils import *

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset("signs")

index = 10
#print(X_train_orig[0, 0, :, :]/255)
#print(Y_train_orig[:, 0])
plt.imshow(X_test_orig[index])
plt.show()
#print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

#print(X_train[index].dtype, Y_train[0].dtype)
#exit()

Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

model = keras.Sequential(
    [
        layers.Conv2D(8, kernel_size=(5, 5), strides = (1, 1), padding='same', name = 'conv0'),
        layers.BatchNormalization(axis = 3, name = 'bn0'),
        layers.Activation('relu'),
        layers.MaxPooling2D((8, 8), strides=(4, 4), name='max_pool0'),

        layers.Conv2D(16, kernel_size=(2, 2), strides = (1, 1), padding='valid', name = 'conv1'),
        layers.BatchNormalization(axis = 3, name = 'bn1'),
        layers.Activation('relu'),
        layers.MaxPooling2D((4, 4), strides=(4, 4), name='max_pool1'),
        
        layers.Flatten(),

        layers.Dense(72, activation='relu', name='fc1'),
        layers.BatchNormalization(axis = 1, name = 'bn2'),
        layers.Activation('relu'),
        layers.Dense(6, name='fc2'),
    ]
)

model(X_train[0:10])
model.summary()

keras.utils.plot_model(model, "cnn_handsigns_tf.png", show_shapes=True)

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x = X_train, y = Y_train, epochs = 50, batch_size = 60)
preds = model.evaluate(X_test, Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))