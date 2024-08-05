import tensorflow as tf
from keras import layers
from keras import Model
#import keras.api.backend as k
#from keras import backend as K
import tensorflow.keras.backend as K

def create_model():
    inputs = layers.Input((64, 64, 1))
    x = layers.Conv2D(96, (11, 11), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (5, 5), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(384, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    pooledOutput = layers.GlobalAveragePooling2D()(x)
    pooledOutput = layers.Dense(1024)(pooledOutput)
    outputs = layers.Dense(128)(pooledOutput)

    model = Model(inputs, outputs)
    return model

feature_extractor = create_model()
feature_extractor.summary()
imgA = layers.Input(shape=(64, 64, 1))
imgB = layers.Input(shape=(64, 64, 1))
featA = feature_extractor(imgA)
featB = feature_extractor(imgB)

def euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = K.sum(K.square(featA - featB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def euclidean_distance2(vectors):
    (featA, featB) = vectors
    sum_squared = tf.reduce_sum(tf.square(featA - featB), axis=1, keepdims=True)
    print(sum_squared)
    #return tf.sqrt(sum_squared)
    return tf.sqrt(tf.maximum(sum_squared, K.epsilon()))

fa = 1*tf.ones((3, 12))
fb = tf.ones((3, 12))
#fa = tf.random.normal([3, 12], mean=6, stddev=0.1, seed = 1)
#fb = tf.random.normal([3, 12], mean=6, stddev=0.1, seed = 1)
print(fa, fb)
result = euclidean_distance2((fa, fb))
print(tf.sigmoid(result))
#exit()

distance = layers.Lambda(euclidean_distance)([featA, featB])
#distance always > 0, why we use the sigmoid here?
outputs = layers.Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])