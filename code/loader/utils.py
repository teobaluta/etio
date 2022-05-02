import os
import math
import numpy as np
import pickle
import torch.nn as nn

# Define loss
criterion_list = {
    "mse": nn.MSELoss(reduction='mean').cuda(),
    "ce": nn.CrossEntropyLoss(reduction="mean").cuda()
}
raw_criterion_list = {
    "mse": nn.MSELoss(reduction="none").cuda(),
    "ce": nn.CrossEntropyLoss(reduction="none").cuda()
}


def nCr(n, r):
    # computing the number of combinations
    f = math.factorial
    return f(n) // f(r) // f(n-r)

"""
Read/Write Files
"""

def read_npy(file_path):
    return np.load(file_path)

def write_txt(file_path, content):
    with open(file_path, "a+") as f:
        f.writelines(content)

def read_txt(file_path):
    with open(file_path) as f:
        data = f.readlines()
    return data

def write_pkl(file_path, content):
    with open(file_path, "wb") as f:
        pickle.dump(content, f)

def read_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

"""
Check/Create Files or Folders
"""
def create_dir(path):
    if not os.path.isdir(path):
        if os.path.isdir(os.path.dirname(path)):
            os.mkdir(path)
        else:
            raise Exception("Failed to create a folder ... %s" % path)
