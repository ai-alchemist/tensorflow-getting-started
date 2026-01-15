# tensorflow-getting-started

This repo contains working up-to-date examples for using TensorFlow.

## 0. Tensors

`tensors_lesson.py` demonstrates a low-level use of tensors.

## 1. Linear Regression

`linear_regression_ttan.py` demonstrates using linear regression to analyze data. An example dataset is provided.

## 2. Neural Nets

`neural_net_train.py` trains a neural net for classifying images. The built-in `keras.datasets.fashion_mnist` is used here.

A custom "handwritten" neural network is used.

`verify_predictions.py` can be used to run predictions on images without having to re-train the neural net.

## 3. Deep Computer Vision

`deep_computer_vision.py` demonstrates training a neural net to classify images. This neural net uses a convolutional neural network for improved classification accuracy.

`finetuning_pretrained_model.py` uses an existing pretrained `MobileNetV2` neural net and finetunes it. This is done by appending a global average pooling layer and a dense layer. Only these last two layers are trained.

## 4. Recurrent Neural Networks

`sentiment_analysis.py` trains a recurrent neural network and uses it to determine whether a string of text is a positive or negative statement.

`text_generation.py` trains a recurrent neural network and uses it to automatically generate more text, similar to a large-language model. The user can optionally provide a string as an input to the RNN, which will affect the RNN's output.

## 5. Q-Learning

`q_learning_gym.py` loads a game environment and uses Q-learning to train a neural net to play a game. In this case, the neural net will move a character from a starting point to a destination while avoiding obstacles.