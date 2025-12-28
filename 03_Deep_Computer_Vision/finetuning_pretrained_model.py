import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# additional imports for finetuning.
keras = tf.keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

print("\n\n-----------------\n\n")

print(tf.__version__)

##############

####################
# Use a pretrained CNN as the start of the model.
# Then, add our own dense layered classifier at the end.
# This can train a good classifier from a relatively small dataset. 
####################

# load the cats_vs_dogs dataset from the modoule tensorflow_datatsets.
# dataset contains (image, label) pairs where images 
# have different dimensions and 3 color channels.

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# creates a function object that we can use to get labels
get_label_name = metadata.features['label'].int2str  

# display 2 images from the dataset
for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))


####################
# Data Preprocessing
####################

# Since the sizes of our images are all different, 
# we need to convert them all to the same size.

IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  """
  returns an image that is reshaped to IMG_SIZE
  """
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

# apply this function to all our images using .map().
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# look at the images again.
for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))


# in the above example, the colors might look weird.
# this probably doesn't affect the training.
# one possible fix for this is to do:
# `plt.imshow( (tf.cast((image + 1.0) * 127.5, tf.uint8)).numpy() )`
# but this probably isn't necessary.

# shuffle and batch the images.
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# compare the shape of an original image vs the new image.
for img, label in raw_train.take(2):
  print("Original shape:", img.shape)

for img, label in train.take(2):
  print("New shape:", img.shape)


########################
# Pretrained Model Setup
########################

########################
# The model we are going to use as the convolutional base for our 
# model is the MobileNet V2 developed at Google.
# This model is trained on 1.4 million images and has 
# 1000 different classes.
#
# We want to use this model but only its convolutional base.
# So, when we load in the model, we'll specify that we don't want
# to load the top (classification) layer.
# We'll tell the model what input shape to expect and to use
# the predetermined weights from imagenet (Google's dataset).
########################
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.summary()

# At this point this base_model will simply output a 
# shape (32, 5, 5, 1280) tensor that is a feature extraction
# from our original (1, 160, 160, 3) image.
# The 32 means that we have 32 layers of differnt filters/features.

for image, _ in train_batches.take(1):
   pass

feature_batch = base_model(image)
print(feature_batch.shape)

# "Freezing the base" refers to disabling the training
# property of a layer.
# It simply means we won’t make any changes to the weights of any
# layers that are frozen during training.
# This is important as we don't want to change the convolutional
# base that already has learned weights.
base_model.trainable = False
base_model.summary()

# Now that we have our base layer setup, we can add the classifier.
# Instead of flattening the feature map of the base layer we will
# use a global average pooling layer that will average the entire
# 5x5 area of each 2D feature map and return to us a
# single 1280 element vector per filter. 
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Finally, we will add the predicition layer that will be a single
# dense neuron.
# We can do this because we only have two classes to predict for.
prediction_layer = keras.layers.Dense(1)

# combine these layers together in a model.
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

model.summary()

########################
# Training the Model
########################

# compile and train the model.
# use a very smalle learning rate to ensure that the model
# doesn't have any major changes made to it.
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# We can evaluate the model right now to see how it does before
# training it on our new images.
initial_epochs = 3
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# Now we train it on our images.
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print("Accuracy: ", acc)

# we can save the model and reload it at anytime in the future.
#model.save("dogs_vs_cats.h5")  
#new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

# h5 format is deprecated, so I will save the model as .keras instead.
model.save("dogs_vs_cats.keras")  
new_model = tf.keras.models.load_model('dogs_vs_cats.keras')

