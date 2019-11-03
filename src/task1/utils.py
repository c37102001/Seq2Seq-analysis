import torch
import pickle


def load_pkl(pkl_path):
    with open(pkl_path, mode='rb') as f:
        obj = pickle.load(f)
    return obj
