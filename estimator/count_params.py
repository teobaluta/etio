import os
import sys

from torch.serialization import save
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from trainer import models, train
import csv

RepeatNum=3
EpochNums = [500, 400]
SchedulerList = ["with_scheduler", "without_scheduler"]
LossDict = {
    "ce": ["cifar10", "cifar100", "mnist"],
    "mse": ["cifar10", "mnist"]
}
LossList = ["mse", "ce"]
DataDict = {
    "cifar10": ["resnet34", "densenet161", "alexnetwol"],
    "cifar100": ["resnet34", "densenet161", "alexnetwol"],
    "mnist": ["mlp_1hidden"]
}
WidthDict = {
    "resnet34": [32,16,8,4,2],
    "densenet161": [32,16,8,4,2],
    "alexnetwol": [256,128,64,32,16],
    "mlp_1hidden": [256,128,64,32,16,8]
}
TrainSizeList = ["5k", "1k"]

def get_param_num(model_path, model_config):
    model = models.MODELS[model_config["model_tag"]](model_config).cuda()
    model.load_state_dict(torch.load(model_path))
    param_num = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_num += parameter.numel()
    return param_num

def parse_args():
    parser = argparse.ArgumentParser(description="Threshold Attacks")
    parser.add_argument(
        "--in_dirs", "-ids", dest="in_dirs", type=str, required=True, help="Input folder"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    in_dir_list = args.in_dirs.split(",")
    for in_dir in in_dir_list:
        if os.path.exists(in_dir) == False:
            raise Exception("Error: Folder does not exist ... %s" % in_dir)
    
    fieldnames = ["Dataset", "Arch", "Width", "TrainSize", "Loss", "Scheduler?", "lr", "EpochNum", "#Params"]
    param_size_list = []
    cnt = 0
    for epoch_num in EpochNums:
        epoch_dir_paths = [os.path.join(in_dir, "epoch_%d" % epoch_num) for in_dir in in_dir_list]
        for data in DataDict:
            for ntrain in TrainSizeList:
                for loss in LossDict:
                    if data not in LossDict[loss]:
                        continue
                    for scheduler in SchedulerList:
                        scheduler_dir_paths = [os.path.join(epoch_dir_path, "%s-%s" % (scheduler, loss)) for epoch_dir_path in epoch_dir_paths]
                        for arch in DataDict[data]:
                            lr = "0.1"
                            if arch == "alexnetwol":
                                lr = "0.01"
                            for width in WidthDict[arch]:
                                cnt += 1
                                tag = True
                                item = [data, arch, width, ntrain, loss, scheduler, lr, epoch_num]
                                for scheduler_dir_path in scheduler_dir_paths:
                                    model_dir_path = os.path.join(
                                        scheduler_dir_path,
                                        "%s-%s-w_%d-ntrain_%s-N_3-less_lr" % (
                                            data, arch, width, ntrain
                                        )
                                    )
                                    if os.path.isdir(model_dir_path) == False:
                                        model_dir_path = os.path.join(
                                            scheduler_dir_path,
                                            "%s-%s-w_%d-ntrain_%s-N_3" % (
                                                data, arch, width, ntrain
                                            )
                                        )
                                    config_path = os.path.join(model_dir_path, "config.ini")
                                    if os.path.exists(config_path) == False:
                                        continue
                                    train_config, model_config = train.load_config(config_path)
                                    model_path = os.path.join(model_dir_path, "models/final_model_1.pkl")
                                    if os.path.exists(model_path) == False:
                                        continue
                                    param_num = get_param_num(model_path, model_config)
                                    item.append(param_num)
                                    tag = False
                                if tag:
                                    item.append("-")
                                param_size_list.append(item)
    out_path = os.path.join(in_dir_list[0], "stats-param_size.csv")
    with open(out_path, mode="w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(fieldnames)
        for item in param_size_list:
            writer.writerow(item)
    


if __name__ == '__main__':
    main()
    