import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import matplotlib.pyplot as plt
import torch
import argparse
from trainer import train, models
import estimate
from tqdm import tqdm
import numpy as np
import shutil

from copy import deepcopy as copy

def parse_args():
    parser = argparse.ArgumentParser(description="Query")
    parser.add_argument(
        "--data_dir", "-dd", dest="data_dir", type=str, required=False, default="", help="Path to the data directory"
    )
    parser.add_argument(
        "--idx_file", "-i", dest="idx_file", type=str, required=True, help="Path to the data index file"
    )
    parser.add_argument(
        "--config_file", "-c", dest="config_file", type=str, required=True, help="Path to the configuration file of the model & the training procedure"
    )
    parser.add_argument(
        "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
    )
    parser.add_argument(
        "--bin_num", "-b", dest="bin_num", type=int, default=10, required=False, help="Number of bins in the histogram"
    )
    parser.add_argument(
        "--train_loss", "-t", dest="train_loss", type=float, default=0.0, required=True, help="Average train loss"
    )
    args = parser.parse_args()
    return args

def get_model_dir(output_dir, model_num):
    final_model_dir = os.path.join(output_dir, "models")
    if not os.path.isdir(final_model_dir):
        print("There is not subfolder named 'models' at %s" % output_dir)
        final_model_dir = output_dir
    for model_id in range(1, model_num+1):
        model_path = os.path.join(final_model_dir, "final_model_%d.pkl" % model_id)
        if not os.path.exists(model_path):
            raise Exception("Error: Cannot find %s" % model_path)
    return final_model_dir

def query_mse(model_config, test_loader, model_path):
    model = models.MODELS[model_config["model_tag"]](model_config).cuda()
    model.load_state_dict(torch.load(model_path))

    model.eval()

    losses = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            
            input_num = labels.size(0)
            labels_onehot = torch.FloatTensor(input_num, int(model_config["out_dim"])).cuda()
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.view(-1, 1).long(), 1)

            outputs = model(inputs)
            losses += list(torch.sum(estimate.raw_criterion_list["mse"](outputs, labels_onehot), dim=1).cpu().numpy())
    return losses

def query_ce(model_config, test_loader, model_path):
    model = models.MODELS[model_config["model_tag"]](model_config).cuda()
    model.load_state_dict(torch.load(model_path))

    model.eval()

    losses = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            outputs = model(inputs)
            losses += list(estimate.raw_criterion_list["ce"](outputs, labels).cpu().numpy())
    return losses


def query(args):
    # load configuration
    train_config, model_config = train.load_config(args.config_file)
    if not os.path.exists(args.idx_file):
        raise Exception("Error: Data idx file does not exist ... %s" % args.idx_file)
    # load test_set
    if len(args.data_dir) == 0:
        args.data_dir = os.path.dirname(args.idx_file)
    data_tag = os.path.basename(args.idx_file).split("-")[1].split("_")[-1]
    train_set, test_set = train.load_complete_data(data_tag, args.data_dir)
    # load idx data
    idx_data = utils.read_npy(args.idx_file)
    test_loader = train.prepare_test_set(data_tag, test_set, idx_data["test_idx_set"], int(train_config["test_batch_size"]))
    # get the model paths
    output_dir = os.path.dirname(args.config_file)
    model_num = idx_data["train_idx_sets"].shape[0]
    model_dir = get_model_dir(output_dir, model_num)
    test_loss_dict = {}
    train_loss_dict = {}
    for model_id in tqdm(range(1, model_num+1), desc="query"):
        model_path = os.path.join(model_dir, "final_model_%d.pkl" % model_id)
        train_loader = train.prepare_test_set(data_tag, copy(train_set), idx_data["train_idx_sets"][model_id-1], int(train_config["test_batch_size"]))
        if train_config["loss"] == "mse":
            test_loss_dict[model_id] = query_mse(model_config, test_loader, model_path)
            train_loss_dict[model_id] = query_mse(model_config, train_loader, model_path)
        elif train_config["loss"] == "ce":
            test_loss_dict[model_id] = query_ce(model_config, test_loader, model_path)
            train_loss_dict[model_id] = query_ce(model_config, train_loader, model_path)
        else:
            raise Exception("Error: Undefined loss function!")
    output_path = os.path.join(output_dir, "test_losses.pkl")
    utils.write_pkl(output_path, test_loss_dict)
    output_path = os.path.join(output_dir, "train_losses.pkl")
    utils.write_pkl(output_path, train_loss_dict)



def plot_tmp(test_loss_path, train_loss_path, bin_num, out_dir):
    test_losses = utils.read_pkl(test_loss_path)
    train_losses = utils.read_pkl(train_loss_path)
    test_list = []
    train_list = []
    for model_id in sorted(list(test_losses.keys())):
        test_list += test_losses[model_id]
        train_list += train_losses[model_id]
    #plt.hist(test_list, density=False, bins=bin_num, color="blue")
    #plt.xlabel("loss")
    #plt.savefig(os.path.join(out_dir, "test_loss.png"))
    #plt.hist(train_list, density=False, bins=bin_num, color="red")
    #plt.xlabel("loss")
    #plt.savefig(os.path.join(out_dir, "train_loss.png"))
    plt.hist([test_list, train_list], bins=bin_num, label=["test", "train"], range=(0,4), density=True)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(out_dir, "train_test_loss.png"))

def main():
    # get arguments
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
  
    output_dir = os.path.dirname(args.config_file)
    test_loss_path = os.path.join(output_dir, "test_losses.pkl")
    train_loss_path = os.path.join(output_dir, "train_losses.pkl")
    #if not os.path.exists(train_loss_path):
    query(args)
    output_dir = os.path.dirname(args.config_file)
    #plot(output_path, args.bin_num, output_dir, args.train_loss)
    plot_tmp(test_loss_path, train_loss_path, args.bin_num, output_dir)

if __name__ == '__main__':
    main()
    
