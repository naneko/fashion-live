from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

from fl.helpers import create_logger, bool_prompt, check_for_file

checkpoint_path = "training/cp.ckpt"
model_path = "training/model.h5"

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
        keras.layers.Dense(128, activation='relu'),  # Fully connected hidden layer
        keras.layers.Dense(10, activation='softmax')  # Output layer
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train(model, epochs):
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

    model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels),
              callbacks=[cp_callback])

    model.save(model_path)

    print('Model Saved')


def app(load_model, load_checkpoint, suppress_prompts, train_epochs):
    if not suppress_prompts:
        if train_epochs != 0:
            if check_for_file(model_path) and not load_model:
                if not bool_prompt('WARNING: Found existing model file. Are you sure you want to overwrite it?', default='no'):
                    print('Exiting...')
                    exit(1)
            if check_for_file(checkpoint_path + '.index') and not load_checkpoint:
                if not bool_prompt('WARNING: Found existing checkpoint file. Are you sure you want to overwrite it?', default='no'):
                    print('Exiting...')
                    exit(1)

    if load_model:
        model = keras.models.load_model(model_path)
    elif load_checkpoint:
        model = create_model()
        model.load_weights(checkpoint_path)
    else:
        model = create_model()

    if train_epochs != 0:
        train(model, train_epochs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("fl")
    parser.add_argument('-t', '--train', help='Train model with TRAIN number of epochs. Set to 0 to skip training.', type=int, default=0)
    parser.add_argument('-y', help='Suppress prompts', action='store_true')
    parser.add_argument('-m', help='If exists, load model', action='store_true')
    parser.add_argument('-c', help='If exists, load checkpoint (will not load checkpoint if -m is present and model exists)', action='store_true')
    args = parser.parse_args()
    app(args.m, args.c, args.y, args.train)
