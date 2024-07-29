import numpy as np
import h5py
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf

import keras
from keras import layers
#from keras import ops

from cnn_utils import *

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset("happy")

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

class EmotionModel(tf.keras.Model):
      def __init__(self):
        super().__init__()
        self.conv2d = Conv2D(32, (3, 3), strides = (1, 1), padding='same', name = 'conv0')
        self.bn = BatchNormalization(axis = 3, name = 'bn0')
        self.act1 = Activation('relu')
        self.pooling = MaxPooling2D((2, 2), name='max_pool')
        self.flt = Flatten()
        self.dense = Dense(1, activation='sigmoid', name='fc')
    
      def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.bn(x)
        x = self.act1(x)
        x = self.pooling(x)
        x = self.flt(x)
        return self.dense(x)

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]


    Returns:
    model -- a Model() instance in Keras
    """
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    # X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides = (1, 1), padding='same', name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model

model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), strides = (1, 1), padding='same', name = 'conv0'),
        layers.BatchNormalization(axis = 3, name = 'bn0'),
        layers.Activation('relu'),
        MaxPooling2D((2, 2), name='max_pool'),
        Flatten(),
        layers.Dense(1, activation='sigmoid', name='fc'),
    ]
)  # No weights at this stage!

model(X_train[0:10])
model.summary()

keras.utils.plot_model(model, "cnn_emotion_detection.png", show_shapes=True)
#exit()

#print(X_train[1:].shape, X_train[0,:].shape)
#model = HappyModel(X_train[0,:].shape)
#model = EmotionModel()
#model(X_train)
#model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 60)
preds = model.evaluate(X_test, Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))