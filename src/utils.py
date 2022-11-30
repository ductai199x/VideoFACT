import os

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


def get_all_files(path, prefix="", suffix="", contains=""):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    files = []
    for pre, dirs, basenames in os.walk(path):
        for name in basenames:
            if name.startswith(prefix) and name.endswith(suffix) and contains in name:
                files.append(os.path.join(pre, name))
    return files


def rand_split(x, r):
    return random_split(x, [int(len(x) * r), len(x) - int(len(x) * r)])


def normal_split(x, r):
    return train_test_split(range(len(x)), train_size=r, shuffle=False)
