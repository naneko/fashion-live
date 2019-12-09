import datetime
import logging
import sys


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
    except IOError:
        exists = False
    finally:
        f.close()

    return exists


