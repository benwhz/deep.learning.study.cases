import tensorflow as tf
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Input(shape = (3, 4)),
        layers.Dense(2, activation='relu'),
        layers.Dense(3, activation='relu'),
        layers.Dense(4)
    ]
)
#x = tf.ones((3, 3))
#y = model()
model.summary()
#print(y)
print(100*'-')

initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
initial_model.summary()
x = tf.ones((1, 250, 250, 3))
y = initial_model(x)
for layer in initial_model.layers:
    print(layer.output)
#feature_extractor.summary()
#features = feature_extractor(x)
#print(features)
print(y)
exit()
print(100*'-')

model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB images
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))

# Can you guess what the current output shape is at this point? Probably not.
# Let's just print it:
model.summary()

# The answer was: (40, 40, 32), so we can keep downsampling...

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

# And now?
model.summary()

# Now that we have 4x4 feature maps, time to apply global max pooling.
model.add(layers.GlobalMaxPooling2D())

# Finally, we add a classification layer.
model.add(layers.Dense(10))

model.summary()

