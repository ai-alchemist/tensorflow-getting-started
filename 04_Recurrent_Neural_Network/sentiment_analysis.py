import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# Additional imports needed for RNN tutorial.
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

print("\n\n-----------------\n\n")

print(tf.__version__)

########################################

####################
# This dataset contains 25,000 reviews from IMDB where each one is
# already preprocessed and has a label as either positive or negative.
# Each review is encoded by integers that represents how common a word
# is in the entire dataset. For example, a word encoded by the integer
# 3 means that it is the 3rd most common word in the dataset.
#
# This guide is based on the tutorial from:
# https://www.tensorflow.org/text/tutorials/text_classification_rnn
####################

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

# Let's look at one review
train_data[1]

####################
# We cannot pass different length data into our neural network.
# We must make each review the same length.
# If the review is greater than 250 words, then trim off the extra words.
# if the review is less than 250 words add the necessary amount of 
# 0s to make it equal to 250.
####################

# user keras's built-in function to do this.
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

####################
# Creating the Model
####################

# use a word embedding layer as the first layer in the model
# and add an LSTM layer afterwards that feeds into a dense node
# to get the predicted sentiment.
#
# 32 stands for the output dimension of the vectors generated
# by the embedding layer.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

####################
# Training the Model
####################

# compile and train the model.
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# evaluate the model using the test data.
results = model.evaluate(test_data, test_labels)
print(results)


####################
# Making Predictions
####################

# Since our reviews are encoded, we'll need to convert
# any review that we write into that form so the network
# can understand it. 
# To do that, we'll load the encodings from the dataset
# and use them to encode our own data.

word_index = imdb.get_word_index()

def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)


# let's make a decode function
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]
  
print(decode_integers(encoded))

# now time to make a prediction
def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred) 
  print(result[0])

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

neutral_review = "The movie was okay."
predict(neutral_review)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)








