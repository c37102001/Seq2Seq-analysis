import unicodedata
import re
import pickle
import torch

SOS_token = 0
EOS_token = 1


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Lang':
            from preprocess import Lang
            return Lang
        return super().find_class(module, name)


def load_pkl(pkl_path):
    with open(pkl_path, mode='rb') as f:
        obj = pickle.load(f)
    return obj

