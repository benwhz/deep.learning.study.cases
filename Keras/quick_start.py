import numpy as np
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
s = slice(0, 1000)
# 
x_train = x_train[s]
y_train = y_train[s]
x_test = x_test[s]
y_test = y_test[s]

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("x_train shape:", x_train.shape)

# Make sure images have shape (28, 28, 1), 
# 1 is channel, data format is channel_last (keras default setting, config in `~/.keras/keras.json`)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Model parameters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 10

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
'''
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]


model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
score = model.evaluate(x_test, y_test, verbose=0)
'''

class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, name=None):
        super().__init__(name=name)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=keras.initializers.GlorotNormal(),
            name="kernel",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer=keras.initializers.Zeros(),
            name="bias",
            trainable=True,
        )
        
        self.output_shape = list(input_shape)
        self.output_shape[1] = self.units

    def compute_output_shape(self, *args, **kwargs):
        return self.output_shape

    def call(self, inputs):
        # Use Keras ops to create backend-agnostic layers/metrics/etc.
        x = keras.ops.matmul(inputs, self.w) + self.b
        return self.activation(x)
    
class MyDropout(keras.layers.Layer):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        self.rate = rate
        # Use seed_generator for managing RNG state.
        # It is a state element and its seed variable is
        # tracked as part of `layer.variables`.
        self.seed_generator = keras.random.SeedGenerator(1337)
        
    def build(self, input_shape):
        self.output_shape = input_shape

    def compute_output_shape(self, *args, **kwargs):
        return self.output_shape
    
    def call(self, inputs):
        # Use `keras.random` for random ops.
        return keras.random.dropout(inputs, self.rate, seed=self.seed_generator)
    
class MyModel(keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_base = keras.Sequential(
            [
                #keras.layers.Input(shape=input_shape),
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                keras.layers.GlobalAveragePooling2D(),
            ]
        )
        self.dp = MyDropout(0.5)
        self.dense = MyDense(num_classes, activation="softmax")

    def build(self, input_shape):
        self.conv_base.build(input_shape)
        self.dp.build(self.conv_base.output_shape)
        self.dense.build(self.dp.output_shape)

    def call(self, x):
        x = self.conv_base(x)
        x = self.dp(x)
        return self.dense(x)
    
model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

model.build((None, 28, 28, 1))
model.summary()

'''
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=10,  # For speed
    validation_split=0.15,
)
'''

import torch

# Create a TensorDataset
train_torch_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train), torch.from_numpy(y_train)
)
val_torch_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_test), torch.from_numpy(y_test)
)

# Create a DataLoader
train_dataloader = torch.utils.data.DataLoader(
    train_torch_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_torch_dataset, batch_size=batch_size, shuffle=False
)

model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
#model.fit(train_dataloader, epochs=10, validation_data=val_dataloader)

import tensorflow as tf

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
model.fit(train_dataset, epochs=10, validation_data=test_dataset)