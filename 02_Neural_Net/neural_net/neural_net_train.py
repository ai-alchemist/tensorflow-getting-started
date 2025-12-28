import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print("\n\n-----------------\n\n")

print(tf.__version__)

##############

# load dataset
fashion_mnist = keras.datasets.fashion_mnist

# split into training and testing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# show the training data.
# If the result is "(60000), 28, 28",
# then that means there are 60,000 28x28 images for training.
# In this example, one image contains 784 pixels.
print('train_images.shape:', train_images.shape)

# print the type of the data.
print('type(train_images):', type(train_images))

# let's have a look at one pixel
# "0, 23, 23" means "row 23, column 23 of the first image".
print('train_images[0,23,23]:', train_images[0,23,23])

# let's have a look at the first 10 training labels
print('train_labels[:10]:', train_labels[:10])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show what one of these images look like.
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# preprocess the data.
# scale all of the pixel values between 0 and 1.
# this will make it easier for the model to process the values.
# apply this to the training AND testing images.
train_images = train_images / 255.0
test_images = test_images / 255.0

##############

# define the neural net.
# use a "sequential" model that passes values from left to right.
# "Flatten" takes the 28x28 shape and converts it into a 
# 1-dimensional value.
# In other words, "use 784 input neurons".
# "Dense" means that every neuron in the previous layer is connected to
# every neuron in the current layer.

# 128 is chosen as the number of "hidden" neurons. 
# The exact value depends on the situation.
# Sometimes, it's a little less than the input layer.
# Sometimes, it will be half of the input layer.
# Sometimes, it's something else.
# Activation function is "rectified linear unit".

# Use 10 output neurons.
# Use as many neurons as there are classes.
# That's why 10 output neurons are used here.
# Use "softmax" activation function.
# Softmax makes sure that sum of all output neurons is 1.
# Constrains output of neurons to between 0 and 1.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

# compile the neural net.
# Use the "adam" optimizer. It performs the "gradient descent".
# Use the "sparse_categorical_crossentropy" loss function.
# Desired metrics is accuracy.

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# we pass the data, labels and epochs and watch the magic!
# "fitting" is the training process.
model.fit(train_images, train_labels, epochs=10)

# evaluate the actual accuracy of the model.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

# nominal accuracy is 91%. Actual test accuracy is 88%.
# this is "overfitting". This is not good for analyzing new data.
# we want highest accuracy possible ON NEW DATA.
# more epochs does not necessarily yield better results.
print('Test accuracy:', test_acc)

# make predictions.
# function accepts an array of images that we want to "predict"
# the class of (T-shirt, Trouser, etc).
# model.predict always expects an array, even for single items.
# to "predict" a single item, do "[test_images[0]]"
predictions = model.predict(test_images)

# show predictions for the first image.
# "probability distribution" that was calculated for output layer.
print('Predictions:', predictions[0])

# print index of maximum value in the list of predictions
# for the first image.
print('Max Prediction:', np.argmax(predictions[0]))

# print the predicted class of the first image as a string.
print('Predicted class:', class_names[np.argmax(predictions[0])])

print('Test Label:', test_labels[0])

# show what the first image looks like.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
