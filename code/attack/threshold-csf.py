"""
Threshold Attack
Reference: Adversary 1 from https://arxiv.org/pdf/1709.01604.pdf
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RandomSeedForData = "0"
import csv
import torch
import argparse
from estimator import estimate
from trainer import models,train
import loader.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm

ClassNums = {
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100
}
TrainSizeMap = {
     "1k" : "1k", # 1000,
     "5k" : "5k", #5000,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Threshold Attacks")
    parser.add_argument(
        "--in_dir", "-id", dest="in_dir", type=str, required=True, help="Input folder"
    )
    parser.add_argument(
        "--out_dir", "-od", dest="out_dir", type=str, required=False, default="", help="Output folder"
    )
    parser.add_argument(
        "--stats_file", "-s", dest="stats_file", type=str, required=True, help="Path to the stats file"
    )
    parser.add_argument(
        "--idx_file", "-i", dest="idx_file", type=str, required=True, help="Path to the index file"
    )
    parser.add_argument(
        "--data_dir", "-d", dest="data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--model_num_per_repeat", "-n", dest="model_num_per_repeat", type=int, required=True, help="Number of models to train per trial"
    )
    parser.add_argument(
        "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
    )
    parser.add_argument(
        "--repeat_num", dest="repeat_num", type=int, required=True, help="The number of trials."
    )
    args = parser.parse_args()
    return args

def get_model_and_train_params(input_dir):
    tmp = os.path.basename(input_dir).split("-")
    tmp2 = os.path.basename(os.path.dirname(input_dir)).split("-")
    wd=os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(input_dir)))).split("-")[1].split("_")[1] # weight decay
    wd_value = 0.005 if wd == "5e3" else 0.0005 if wd == "5e4" else 0
    if wd_value == 0: raise Exception("Can't get Weight Decay Value from input_dir name")
    try:
        params = {
            "Dataset": tmp[0],
            "Arch": tmp[1],
            "Width": int(tmp[2].split("_")[-1]),
            "TrainSize": TrainSizeMap[tmp[3].split("_")[-1]],
#            "SplitNum": int(tmp[4].split("_")[-1]),
            "Scheduler?": tmp2[0],
            "Loss": tmp2[1],
            "EpochNum": int(os.path.basename(os.path.dirname(os.path.dirname(input_dir))).split("_")[-1]),
            "WeightDecay": wd_value
        }
    except:
        raise Exception("Error: Failed to get the parameters of the training algorithm and the model ... %s" % input_dir)
    else:
        pass
    return params

def compute_loss_ce(model, loader, class_num):
    model.eval()
    individual_loss_array = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()

            outputs = model(inputs)
            individual_loss_array += list(estimate.raw_criterion_list["ce"](outputs, labels).cpu().numpy())
    return individual_loss_array

def compute_loss_mse(model, loader, class_num):
    model.eval()
    individual_loss_array = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            input_num = labels.size(0)
            labels_onehot = torch.FloatTensor(input_num, class_num).cuda()
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.view(-1, 1).long(), 1)
            outputs = model(inputs)
            individual_loss_array += list(torch.sum(estimate.raw_criterion_list["mse"](outputs, labels_onehot), dim=1).cpu().numpy())
    return individual_loss_array

def compute_acc(loss_list, threshold, tag):
    losses = np.asarray(loss_list)
    if tag == "test":
        acc = len(np.where(losses > threshold)[0]) * 1.0 / len(losses)
    elif tag == "train":
        acc = len(np.where(losses <= threshold)[0]) * 1.0 / len(losses)
    else:
        raise Exception("Error: Undefined tag ... %s" % tag)
    return acc

def threshold_attack(compute_loss_fn, params, test_loader, model_paths, model_config, train_set, train_idx_sets, batch_size, threshold, repeat_id):
    class_num = int(model_config["out_dim"])
    model_num = len(model_paths)
    acc_dict = {"train": [], "test": []}
    for model_id in tqdm(range(1, model_num+1), desc="Repeat %d" % repeat_id):
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        model.load_state_dict(torch.load(model_paths[model_id-1]))
        test_loss = compute_loss_fn(model, test_loader, class_num)
        test_acc = compute_acc(test_loss, threshold, "test")
        acc_dict["test"].append(test_acc)

        target_model_id = int(os.path.basename(model_paths[model_id-1]).split(".")[0].split("_")[-1])
        train_set_copy = deepcopy(train_set)
        train_loader = train.prepare_test_set(params["Dataset"], train_set_copy, train_idx_sets[target_model_id-1], batch_size)
        train_loss = compute_loss_fn(model, train_loader, class_num)
        train_acc = compute_acc(train_loss, threshold, "train")
        acc_dict["train"].append(train_acc)
        print("Model Id: ", model_id)
    return acc_dict


def get_idx_file(data_dir, setup_dir_path):
    setting = os.path.basename(setup_dir_path).split("-")
    data_tag = setting[0]
    if setting[3][:7] == "ntrain_" and setting[3][-1] == "k":
        train_subset_size = int(setting[3][:-1].split("_")[1]) * 1000
    else:
        raise Exception("Error: Cannot parse the setup ... %s" % setup_dir_path)
    if setting[4][:2] == "N_":
        split_num = int(setting[4].split("_")[1])
    else:
        raise Exception("Error: Cannot parse the setup ... %s" % setup_dir_path)
    idx_file = os.path.join(
        data_dir,
        "prepare_data-data_%s-ntrial_%d-train_size_%d-test_size_10000-strategy_disjoint_split-seed_%s.npz" % (
            data_tag, split_num, train_subset_size, RandomSeedForData
        )
    )
    if not os.path.exists(idx_file):
        raise Exception("Error: Cannot find the data idx file ... %s" % idx_file)
    return idx_file, data_tag, train_subset_size, split_num

TrainSize = {
    "mnist": 60*1000,
    "cifar10": 50*1000,
    "cifar100": 50*1000,
}

def main():
    data_dir = sys.argv[1]
    setup_dir_path = sys.argv[2]
    GPU_ID = sys.argv[3]
    assert(int(GPU_ID)>= 0)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    model_dir = os.path.join(setup_dir_path, "models")
    config_file = os.path.join(setup_dir_path, "config.ini")
    out_path = os.path.join(setup_dir_path, "results-threshold_csf.pkl")
    if os.path.exists(out_path):
        print("Results for threshold attack are ready ... %s" % out_path)
        acc_dict = utils.read_pkl(out_path)
        print("Threshold Dict:", acc_dict)
    else:
        if os.path.exists(config_file) == False:
            raise Exception("Error: Cannot find the file ... %s" % config_file)
        train_config, model_config = train.load_config(config_file)
        params = get_model_and_train_params(setup_dir_path)
        stat_path = os.path.join(setup_dir_path, "stats.pkl")
        stats = utils.read_pkl(stat_path)
        avg_test_loss, avg_test_acc, avg_test_var, avg_test_bias2, avg_train_loss, avg_train_acc, avg_train_var, avg_train_bias2 = estimate.get_stats(stats)
        if params["Loss"] == "mse":
            compute_loss_fn = compute_loss_mse
        elif params["Loss"] == "ce":
            compute_loss_fn = compute_loss_ce
        else:
            raise Exception("Error: Undefined loss ... %s" % params["Loss"])
        # load data
        train_set, test_set = train.load_complete_data(params["Dataset"], data_dir)
        # load idx data
        idx_file, data_tag, train_subset_size, split_num = get_idx_file(data_dir, setup_dir_path)
        idx_data = utils.read_npy(idx_file)
        test_loader = train.prepare_test_set(params["Dataset"], test_set, idx_data["test_idx_set"], int(train_config["test_batch_size"]))
        
        # prepare models
        model_num_per_repeat = int(TrainSize[data_tag]/train_subset_size)
        total_model_num = model_num_per_repeat * split_num
        model_path_array = estimate.get_model_paths(model_dir,model_num_per_repeat, split_num)

        # perform threshold attacks
        repeat_id = 0
        acc_dict = {}
        for model_paths in model_path_array:
            acc_dict["repeat_%d" % repeat_id] = threshold_attack(compute_loss_fn, params, test_loader, model_paths, model_config, train_set, idx_data["train_idx_sets"], int(train_config["test_batch_size"]), avg_train_loss, repeat_id)
            print("Repeat Id: ", repeat_id)
            repeat_id += 1
        acc_dict["threshold"] = avg_train_loss
        utils.write_pkl(out_path, acc_dict)


if __name__ == '__main__':
    main()
