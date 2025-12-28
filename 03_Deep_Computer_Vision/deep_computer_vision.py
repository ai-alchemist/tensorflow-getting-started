import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

print("\n\n-----------------\n\n")

print(tf.__version__)

##############

####################
# The problem we will consider here is classifying 10 different 
# everyday objects. 
# The dataset we will use is built into tensorflow and called the 
# CIFAR Image Dataset. It contains 60,000 32x32 color images with 6000 
# images of each class. 
####################

# load and split dataset.
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images

plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

####################
# A common architecture for a CNN is a stack of Conv2D and MaxPooling2D 
# layers followed by a few denesly connected layers. 
# The idea is that the stack of convolutional and maxPooling layers 
# extract the features from the image. 
# Then these features are flattened and fed to densely connected layers 
# that determine the class of an image based on the presence of features.
####################

# start by building the Convolutional Base.
model = models.Sequential()

# The input shape of our data will be 32, 32, 3 
# and we will process 32 filters of size 3x3 over our input data. 
# We will also apply the activation function relu to the output
# of each convolution operation.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# Layer 2 will perform the max pooling operation using 2x2 samples and
# a stride of 2.
model.add(layers.MaxPooling2D((2, 2)))

# The next set of layers do very similar things but take as input the
# feature map from the previous layer. 
# They also increase the frequency of filters from 32 to 64. 
# We can do this as our data shrinks in spacial dimensions as it passed
# through the layers, meaning we can afford (computationally)
# to add more depth.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# now add the dense layers to the model.
# these will be used to classify the extracted features.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()


####################
# Training
####################

# train and compile the model using the recommended 
# hyper parameters from tensorflow.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(test_images, test_labels))

# evaluate the model.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

