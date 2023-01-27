from .save import dump, load, create_if_not_exist
import os


def join(*paths):
    return os.path.join(*paths)
