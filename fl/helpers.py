import datetime
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np


def create_logger(name=__name__, level=logging.INFO, tag=''):
    """
    Create MachLash Logger

    :param name: Logger name
    :param level: Console output logger level
    :param tag: String to include in filename of logger output
    :return: Logger object
    """
    logger = logging.Logger(name)
    logger.setLevel(logging.DEBUG)
    format_str = '[%(asctime)s] [%(levelname)s] %(message)s'
    date_format = '%H:%M:%S'
    plain_formatter = logging.Formatter(format_str, date_format)
    try:
        import colorlog
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)
    except:
        formatter = plain_formatter
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    date_now = datetime.datetime.now()
    date = date_now.strftime('%Y-%m-%d--%I-%M-%S-%p')
    fh = logging.FileHandler(tag + '-' + date + '.log')
    fh.setFormatter(plain_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger


def bool_prompt(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def check_for_file(path):
    try:
        f = open(path)
        exists = True
        f.close()
    except IOError:
        exists = False

    return exists


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_image_live(splt, i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    splt.grid(False)
    splt.set_xticks([])
    splt.set_yticks([])

    splt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue'

    splt.set_xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                100*np.max(predictions_array)),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_value_array_live(splt, i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    splt.grid(False)
    splt.set_xticks(range(10))
    splt.set_yticks([])
    thisplot = splt.bar(range(10), predictions_array, color="#777777")
    splt.set_ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('blue')
