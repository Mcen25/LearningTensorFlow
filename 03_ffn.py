import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# (60000, 28, 28) (60000,)
# 60,000 images, 28 x 28 pixels, with 60,000 labels

#normalize the data 
x_train, x_test = x_train / 255.0, x_test / 255.0

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(x_train[i], cmap='gray')
# plt.show()

# Dense layer is a fully connected layer

# activation functions improves the training, introduces non-linearity!
# relu is the most popular activation function
# softmax is used for multi-class classification
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())

#loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

#predictions
prob_model = keras.models.Sequential([
    model, 
    keras.layers.Softmax()
])

# predictions = prob_model.predict(x_test)
# pred0 = predictions[0]
# print(pred0)
# label0 = np.argmax(pred0)
# print(label0)

# model and softmax
predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

pred05 = predictions[0:5]
print(pred05.shape)
label05 = np.argmax(pred05, axis=1)
print(label05)