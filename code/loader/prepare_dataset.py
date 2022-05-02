import sys
sys.path.append("../")

import os
import loader.utils as utils
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import argparse
from hashlib import sha256

class FashionMNISTLoader(object):
    def __init__(self, data_path):
        # Size(Training Set): 60000
        # Size(Training Set): 10000
        self.tag = "fmnist"
        self.data_path = data_path

    def download(self):
        download_tag = True
        data_path = os.path.join(self.data_path, "FashionMNIST")
        if os.path.isdir(data_path):
            download_tag = False
        train_set = torchvision.datasets.FashionMNIST(root=self.data_path, train=True, download=download_tag)
        test_set = torchvision.datasets.FashionMNIST(root=self.data_path, train=False, download=download_tag)
        return train_set, test_set

class MNISTLoader(object):
    def __init__(self, data_path):
        # Size(Training Set): 60000
        # Size(Training Set): 10000
        self.tag = "mnist"
        self.data_path = data_path

    def download(self):
        download_tag = True
        data_path = os.path.join(self.data_path, "MNIST")
        if os.path.isdir(data_path):
            download_tag = False
        train_set = torchvision.datasets.MNIST(root=self.data_path, train=True, download=download_tag)
        test_set = torchvision.datasets.MNIST(root=self.data_path, train=False, download=download_tag)
        return train_set, test_set

class Cifar10Loader(object):
    def __init__(self, data_path):
        # Size(Training Set): 50000
        # Size(Training Set): 10000
        self.tag = "cifar10"
        self.data_path = data_path

    def download(self):
        download_tag = True
        data_path = os.path.join(self.data_path, "cifar-10-batches-py")
        if os.path.isdir(data_path):
            download_tag = False
        train_set = torchvision.datasets.CIFAR10(root=self.data_path, train=True, download=download_tag)
        test_set = torchvision.datasets.CIFAR10(root=self.data_path, train=False, download=download_tag)
        return train_set, test_set

class Cifar100Loader(object):
    def __init__(self, data_path):
        # Size(Training Set): 50000
        # Size(Training Set): 10000
        self.tag = "cifar100"
        self.data_path = data_path
    
    def download(self):
        download_tag = True
        data_path = os.path.join(self.data_path, "cifar-100-python")
        if os.path.isdir(data_path):
            download_tag = False
        train_set = torchvision.datasets.CIFAR100(root=self.data_path, train=True, download=download_tag)
        test_set = torchvision.datasets.CIFAR100(root=self.data_path, train=False, download=download_tag)
        return train_set, test_set

def random_sample_w_collision(train_size, set_size, model_num, strategy):
    train_idx_array = []
    for _ in tqdm(range(model_num), desc="[PrepTrainSets-%s]" % strategy):
        idx_list = np.arange(train_size)
        np.random.shuffle(idx_list)
        train_idx_array.append(idx_list[: set_size])
    return train_idx_array

def random_sample_wo_collision(train_size, set_size, model_num, strategy):
    max_model_num = utils.nCr(train_size, set_size)
    if (max_model_num <= model_num):
        raise Exception("ERROR: Cannot train so many models with different sets ... #Models/Max(#Models): %d/%d" % (model_num,max_model_num))
    print("#Models/Max(#Models): %d/%d" % (model_num, max_model_num))
    cnt = 0
    hash_list = []
    train_idx_array = []
    pbar = tqdm(total = model_num, desc="[PrepTrainSets-%s]" % strategy)
    while(cnt < model_num):
        idx_list = np.arange(train_size)
        np.random.shuffle(idx_list)
        idx_list = idx_list[: set_size]
        hash_value = sha256(idx_list).hexdigest()
        if hash_value not in hash_list:
            train_idx_array.append(idx_list)
            hash_list.append(hash_value)
            cnt += 1
            pbar.update(1)
    pbar.close()
    return train_idx_array

def disjoint_split(train_size, set_size, split_num, strategy):
    train_idx_array = []
    subset_num = train_size // set_size
    for _ in tqdm(range(split_num), desc="[PrepTrainSets-%s]" % strategy):
        idx_list = np.arange(train_size)
        np.random.shuffle(idx_list)
        for set_id in range(subset_num):
            train_idx_array.append(idx_list[set_id*set_size: (set_id+1)*set_size])
    return train_idx_array

TrainSetConstructStrategy = {
    "random-collision": random_sample_w_collision,
    "random-no_collision": random_sample_wo_collision,
    "disjoint_split": disjoint_split
}

class DatasetPreprocessor(object):
    def __init__(self, data_loader):
        # download dataset
        self.data_loader = data_loader
        self.train_set, self.test_set = data_loader.download()
        self.train_size = self.train_set.data.shape[0]
        self.test_size = self.test_set.data.shape[0]

    def prepare_train_sets(self, set_size, trial_num, strategy, rand_seed=0):
        np.random.seed(rand_seed)
        print("Set the random seed to be %d" % rand_seed)
        if (set_size > self.train_size):
            raise Exception("ERROR: Size(training subset) >= Size(training set) ... Size(training subset)/Size(training set): %d/%d" % (set_size, self.train_size))
        train_idx_array = TrainSetConstructStrategy[strategy](self.train_size, set_size, trial_num, strategy)
        train_idx_array = np.asarray(train_idx_array)
        print("#(Unique train Sets): %d" % np.unique(train_idx_array, axis = 0).shape[0])
        return train_idx_array

    def prepare_test_set(self, set_size, rand_seed=0):
        np.random.seed(rand_seed)
        print("Set the random seed to be %d" % rand_seed)
        if (set_size > self.test_size):
            raise Exception("ERROR: Size(testing set) is too small ... #(Required Samples)/Size(testing set): %d/%d" % (set_size, self.test_size))
        idx_list = np.arange(self.test_size)
        np.random.shuffle(idx_list)
        return idx_list[: set_size]

    def save_idx(self, out_dir, train_idx_array, test_idx_array, file_name):
        utils.create_dir(out_dir)
        out_path = os.path.join(out_dir, file_name)
        np.savez(out_path, train_idx_sets=train_idx_array, test_idx_set=test_idx_array)
        print("Saved content in %s.npz" % out_path)

RegisteredDatasets = {
    "cifar100":{
        "loader": Cifar100Loader,
        "train_transformer": transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
            ]
        ),
        "test_transformer": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
            ]
        )
    },
    "cifar10": {
        "loader": Cifar10Loader,
        "train_transformer": transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        ),
        "test_transformer": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
        )
    },
    "mnist": {
        "loader": MNISTLoader,
        "train_transformer": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        ),
        "test_transformer": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    },
    "fmnist": {
        "loader": FashionMNISTLoader,
        "train_transformer": transforms.Compose(
            [
                transforms.ToTensor()
            ]
        ),
        "test_transformer": transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
    }
}

def prepare_dataset(args):
    my_loader = RegisteredDatasets[args.data_tag]["loader"](args.out_dir)
    my_dataset = DatasetPreprocessor(my_loader)
    train_idx_array = my_dataset.prepare_train_sets(args.train_data_size, args.trial_num, args.strategy, args.rand_seed)
    test_idx_array = my_dataset.prepare_test_set(args.test_data_size, args.rand_seed)
    file_name = "prepare_data-data_%s-ntrial_%d-train_size_%d-test_size_%d-strategy_%s-seed_%d" % (
            args.data_tag, args.trial_num, args.train_data_size, args.test_data_size, args.strategy, args.rand_seed
        )
    my_dataset.save_idx(args.out_dir, train_idx_array, test_idx_array, file_name)

def parse_args():
    parser = argparse.ArgumentParser(description="PrepareDataset")
    parser.add_argument(
        "--data_tag", "-d", dest="data_tag", type=str, required=True, choices=list(RegisteredDatasets.keys()), help="Dataset Name"
    )
    parser.add_argument(
        "--out_dir", "-o", dest="out_dir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--train_data_size", "-trds", dest="train_data_size", type=int, required=False, default=1000, help="Number of samples in each training set"
    )
    parser.add_argument(
        "--test_data_size", "-teds", dest="test_data_size", type=int, required=False, default=100, help="Number of samples used for computing E_(x,y)"
    )
    parser.add_argument(
        "--trial_num", "-t", dest="trial_num", type=int, required=False, default=10, help="Number of trials (It can be number of models to train if using 'random' strategy or the number of random splits if using 'disjoint_split' strategy"
    )
    parser.add_argument(
        "--rand_seed", "-rs", dest="rand_seed", type=int, required=False, default=0, help="Seed for randomization"
    )
    parser.add_argument(
        "--strategy", "-s", dest="strategy", type=str, required=False, default="disjoint_split", choices=list(TrainSetConstructStrategy.keys()), help="How to construct training sets for training models"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    prepare_dataset(args)

if __name__ == '__main__':
    main()
    
