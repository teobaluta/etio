"""
Bounded Loss Attack
Reference: Adversary 1 from https://arxiv.org/pdf/1709.01604.pdf
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from trainer import models,train
import loader.utils as utils
from estimator import estimate
import numpy as np
from tqdm import tqdm
import loader.prepare_dataset as prep_data
from copy import deepcopy

B=2.0

def log_args(args, out_dir, attack_type):
    out_path = os.path.join(out_dir, "%s_args.txt" % attack_type)
    in_str = []
    for key in sorted(list(args.keys())):
        in_str.append("%s: %s\n" % (str(key), str(args[key])))
    utils.write_txt(out_path, in_str)

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

def check_member_for_bounded_loss_attack(loss_list):
    prob_list = np.asarray(loss_list)/B
    # The larger the prob is, more likely the sample is from the testing set
    # 0: train, 1: test
    member_labels = [np.random.choice(a=[0,1], size=1, p=[1-prob, prob])[0] for prob in prob_list]
    return member_labels

def check_member_for_threshold_based_attack(loss_list, threshold):
    member_idx_list = np.where(np.asarray(loss_list) < threshold)
    # if the loss is smaller than the given threshold, then this sample is considered as a training sample
    # 0: train, 1: test
    member_labels = np.ones(len(loss_list), dtype=int)
    member_labels[member_idx_list] = 0
    return member_labels

def loss_based_attack(compute_loss_fn, member_fn, log_path, model_config, test_loader, train_set,  model_paths, repeat_id, train_idx_sets, data_tag, batch_size, seeds, threshold):
    class_num = int(model_config["out_dim"])
    model_num = len(model_paths)
    for model_id in range(1, model_num+1):
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        model.load_state_dict(torch.load(model_paths[model_id-1]))
        test_loss = compute_loss_fn(model, test_loader, class_num)
        test_member_labels = member_fn(test_loss, threshold)
        test_attack_acc = 0
        for seed in seeds:
            np.random.seed(seed)
            test_attack_acc += np.sum(test_member_labels)/len(test_member_labels)
        test_attack_acc = test_attack_acc/len(seeds)

        target_model_id = int(os.path.basename(model_paths[model_id-1]).split(".")[0].split("_")[-1])
        train_set_copy = deepcopy(train_set)
        train_loader = train.prepare_test_set(data_tag, train_set_copy, train_idx_sets[target_model_id-1], batch_size)
        train_loss = compute_loss_fn(model, train_loader, class_num)
        train_member_labels = member_fn(train_loss, threshold)
        train_attack_acc = 0
        for seed in seeds:
            np.random.seed(seed)
            train_attack_acc += (len(train_member_labels) - np.sum(train_member_labels))/len(train_member_labels)
        train_attack_acc = train_attack_acc/len(seeds)
        log_str = "%d\t%d\t%f\t%f" % (repeat_id, model_id, train_attack_acc, test_attack_acc)
        print(log_str)
        utils.write_txt(log_path, [log_str + "\n"])

FunctionsForCheckingMembers = {
    "bounded_loss": check_member_for_bounded_loss_attack,
    "threshold": check_member_for_threshold_based_attack
}


def parse_args():
    parser = argparse.ArgumentParser(description="Loss-based Attack Without Training Shadow Models")
    parser.add_argument(
        "--data_tag", "-d", dest="data_tag", type=str, required=True, choices=list(prep_data.RegisteredDatasets.keys()), help="Dataset Name"
    )
    parser.add_argument(
        "--data_dir", "-dd", dest="data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--idx_file", "-i", dest="idx_file", type=str, required=True, help="Path to the data index file"
    )
    parser.add_argument(
        "--config_file", "-c", dest="config_file", type=str, required=True, help="Path to the configuration file of the model & the training procedure"
    )
    parser.add_argument(
        "--model_num_per_repeat", "-n", dest="model_num_per_repeat", type=int, default=6, help="Number of models to train per trial"
    )
    parser.add_argument(
        "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
    )
    parser.add_argument(
        "--seeds", "-s", dest="seeds", type=str, default="0,1,2", required=False, help="The random seeds for performing bounded loss attack."
    )
    parser.add_argument(
        "--attack_type", "-a", dest="attack_type", type=str, required=True, choices=list(FunctionsForCheckingMembers.keys()), help="The kind of attack to perform"
    )
    parser.add_argument(
        "--threshold", "-t", dest="threshold", type=float, required=False, default=0.0, help="The threshold for threshold-based attack (no need to specify it for bounded loss attack)."
    )
    parser.add_argument(
        "--repeat_num", dest="repeat_num", type=int, required=True, help="Number of trials (N=3)."
    )
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # load config
    if not os.path.exists(args.config_file):
        raise Exception("Error: Configuration file does not exist ... %s" % args.config_file)
    train_config, model_config = train.load_config(args.config_file)

    if train_config["loss"] == "mse":
        compute_loss_fn = compute_loss_mse
        member_fn = FunctionsForCheckingMembers[args.attack_type]
    elif train_config["loss"] == "ce":
        compute_loss_fn = compute_loss_ce
        if args.attack_type == "bounded_loss":
            raise Exception("Error: Unknown bound for the loss for %s attack" % args.attack_type)
        member_fn = FunctionsForCheckingMembers[args.attack_type]
    else:
        raise Exception("Error: Please define whether to add a softmax layer in the model and which function to use for computing loss")

    # load data
    if not os.path.exists(args.idx_file):
        raise Exception("Error: Data idx file does not exist ... %s" % args.idx_file)
    train_set, test_set = train.load_complete_data(args.data_tag, args.data_dir)
    # load idx data
    idx_data = utils.read_npy(args.idx_file)

    model_dir = os.path.dirname(args.config_file)
    log_args(vars(args), model_dir, args.attack_type)
    # prepare test & train loader
    test_loader = train.prepare_test_set(args.data_tag, test_set, idx_data["test_idx_set"], int(train_config["test_batch_size"]))
    model_path_array = estimate.get_model_paths(model_dir, args.model_num_per_repeat, args.repeat_num)
    repeat_id = 0
    # prepare random seeds & check the repeatness
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if args.attack_type == "bounded_loss":
        log_path = os.path.join(model_dir, "%s-seed_%s.txt" % (args.attack_type, "_".join([str(seed) for seed in seeds])))
        if len(seeds) < 2:
            raise Exception("Error: Please repeat the attack for > 1 times at least for bounded loss attack")
    elif args.attack_type == "threshold":
        log_path = os.path.join(model_dir, "%s-seed_%s-t_%f.txt" % (args.attack_type, "_".join([str(seed) for seed in seeds]), args.threshold))
        seeds = seeds[:1]
    else:
        raise Exception("Error: Please define whether to repeat %s attack" % args.attack_type)

    if os.path.exists(log_path):
        os.remove(log_path)
    log_str = "#repeat\t#model\tTrain Acc\tTest Acc"
    print(log_str)
    utils.write_txt(log_path, [log_str+"\n"])

    for model_paths in model_path_array:
        repeat_id += 1
        loss_based_attack(compute_loss_fn, member_fn, log_path, model_config, test_loader, train_set,  model_paths, repeat_id, idx_data["train_idx_sets"], args.data_tag, int(train_config["test_batch_size"]), seeds, args.threshold)
    print("Done!")

if __name__ == '__main__':
    main()
