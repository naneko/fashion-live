from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import cv2

from fl.helpers import *

checkpoint_path = "training/cp.ckpt"
model_path = "training/model.h5"

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),  # Fully connected hidden layer
        keras.layers.Dense(10, activation='softmax')  # Output layer
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train(model, epochs):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)

    model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels),
              callbacks=[cp_callback])

    model.save(model_path)

    predictions = model.predict(test_images)

    show_results(3, 5, predictions)

    print('Model Saved')


def show_results(num_cols, num_rows, predictions):
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


def app(load_model, load_checkpoint, train_epochs):
    if train_epochs != 0:
        if check_for_file(model_path) and not load_model:
            if not bool_prompt('WARNING: Found existing model file. Are you sure you want to overwrite it?', default='no'):
                print('Exiting...')
                exit(1)
            else:
                model = create_model()
        elif check_for_file(model_path) and load_model:
            model = keras.models.load_model(model_path)
        elif check_for_file(checkpoint_path + '.index') and not load_checkpoint:
            if not bool_prompt('WARNING: Found existing checkpoint file. Are you sure you want to overwrite it?', default='no'):
                print('Exiting...')
                exit(1)
            else:
                model = create_model()
        elif check_for_file(checkpoint_path + '.index') and load_checkpoint:
            model = create_model()
            model.load_weights(checkpoint_path)
        else:
            model = create_model()
        train(model, train_epochs)

    elif check_for_file(model_path) and load_model:
        model = keras.models.load_model(model_path)
    elif check_for_file(checkpoint_path + '.index') and load_checkpoint:
        model = create_model()
        model.load_weights(checkpoint_path)
    else:
        print('ERROR: No model could be loaded. Exiting...')
        exit(1)

    predictions = model.predict(test_images)
    # row_count = 0
    # for row in test_images[0]:
    #     row_count += 1
    #     col_count = 0
    #     col_str = ''
    #     for col in row:
    #         col_count += 1
    #         col_str = col_str + 'X'
    #     print(col_str)
    # print('{} x {}'.format(row_count, col_count))

    show_results(3, 5, predictions)

    capture = cv2.VideoCapture(2)

    # SHOW RESULTS CUSTOM #
    # https://stackoverflow.com/questions/42603161/convert-an-image-shown-in-python-into-an-opencv-image#42604008
    num_rows = 1
    num_cols = 1
    num_images = num_rows * num_cols
    fig = plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    while(True):
        if capture.isOpened():
            ret, frame = capture.read()
            w = int(capture.get(3))  # float
            h = int(capture.get(4))  # float
            y = 0
            offset_x = int((w/2)-(h/2))
            roi = frame[y:y+h, offset_x:offset_x+h]
            grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(grey, (28, 28), interpolation=cv2.INTER_AREA)
            invert = cv2.bitwise_not(resize)
            # cv2.imshow('video', invert)
            predictions = model.predict([[invert]])

            # SHOW RESULTS CUSTOM #
            for i in range(num_images):
                splt1 = fig.add_subplot(num_rows, 2 * num_cols, 2 * i + 1)
                plot_image_live(splt1, i, predictions[i], test_labels, [invert], class_names)
                splt2 = fig.add_subplot(num_rows, 2 * num_cols, 2 * i + 2)
                plot_value_array_live(splt2, i, predictions[i], test_labels)
            fig.tight_layout()
            fig.canvas.draw()
            fig.show()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.imshow('plot', img)

            fig.clf()

            if cv2.waitKey(1) == 27:
                break
        else:
            print('ERROR: Camera Access Error\nI was too lazy to make a method of changing the device ID without modifying code. Sorry.')
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("fl")
    parser.add_argument('-t', '--train', help='Train model with TRAIN number of epochs. Set to 0 to skip training.', type=int, default=0)
    parser.add_argument('-m', help='If exists, load model', action='store_true')
    parser.add_argument('-c', help='If exists, load checkpoint (will not load checkpoint if -m is present and model exists)', action='store_true')
    args = parser.parse_args()
    app(args.m, args.c, args.train)
