'''
Get the centroid of the last layer
'''
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import models, train
from copy import deepcopy
from tqdm import tqdm
import loader.utils as utils
from estimator import estimate
import numpy as np
import argparse
import torch
from itertools import product


ComputeTags = ["sorted_3", "sorted_10", "origin"]
RepeatNum = 3
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
    "resnet34": [32, 16, 8, 4, 2],
    "densenet161": [32, 16, 8, 4, 2],
    "alexnetwol": [256, 128, 64, 32, 16],
    # "mlp_1hidden": [256,128,64,32,16,8]
}
TrainSizeList = ["5k", "1k"]
ModelNumPerRepeat = {"1k": 50, "5k": 10}

TRAIN_SIZE_DICT = {"1k": "1000", "5k": "5000", "10k": "10000"}


def get_last_layer_output(model, loader):
    model.eval()
    last_layer_output = None
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()

            outputs = model(inputs).cpu().numpy()
            if last_layer_output is None:
                last_layer_output = outputs
            else:
                last_layer_output = np.concatenate(
                    (last_layer_output, outputs))
    return last_layer_output


def get_last_layer_for_each_setup(repeat_id, model_paths, out_dir, model_config, test_loader, train_set, train_idx_sets, batch_size, dataset):
    model_num = len(model_paths)
    for model_id in tqdm(range(1, model_num+1), desc="Repeat %d" % repeat_id):
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        model.load_state_dict(torch.load(model_paths[model_id-1]))
        test_outputs = get_last_layer_output(model, test_loader)

        target_model_id = int(os.path.basename(
            model_paths[model_id-1]).split(".")[0].split("_")[-1])
        train_set_copy = deepcopy(train_set)
        train_loader = train.prepare_test_set(
            dataset, train_set_copy, train_idx_sets[target_model_id-1], batch_size)
        train_outputs = get_last_layer_output(model, train_loader)

        tmp = os.path.basename(model_paths[model_id-1]).split(".")[0]
        out_path = os.path.join(out_dir, f"{tmp}-dataset.npz")
        np.savez(out_path, train=train_outputs, test=test_outputs)


def sorted_all_centroid(inputs, dim):
    sorted_inputs = np.sort(inputs)
    dim_num = sorted_inputs.shape[-1]
    centroid = []
    for dim_no in range(dim_num):
        centroid.append(np.mean(sorted_inputs[:, dim_no]))
    return np.asarray(centroid)


def sorted_maxn_centroid(inputs, maxn=1):
    sorted_inputs = np.sort(inputs)
    centroid = []
    for dim_no in range(1, maxn+1):
        centroid = [np.mean(sorted_inputs[:, -dim_no])] + centroid
    return np.asarray(centroid)


def nonsorted_all_centroid(inputs, dim):
    dim_num = inputs.shape[-1]
    centroid = []
    for dim_no in range(dim_num):
        centroid.append(np.mean(inputs[:, dim_no]))
    return np.asarray(centroid)


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


def compute_train_test_centroid(out_dir, compute_tag):
    if compute_tag == "origin":
        compute_fn = nonsorted_all_centroid
        dim = None
    elif compute_tag == "sorted":
        compute_fn = sorted_all_centroid
        dim = None
    elif compute_tag[:7] == "sorted_":
        compute_fn = sorted_maxn_centroid
        dim = int(compute_tag.split("_")[-1])
    else:
        raise Exception(f"Error: Undefined compute_tag ... {compute_tag}")
    file_list = os.listdir(out_dir)
    distance_list = []
    for file_name in tqdm(file_list, desc=f"CentroidDistance({compute_tag})"):
        if "dataset" in file_name:
            file_path = os.path.join(out_dir, file_name)
            last_layer_outputs = np.load(file_path)
            train_outputs = last_layer_outputs["train"]
            test_outputs = last_layer_outputs["test"]
            train_centroid = compute_fn(train_outputs, dim)
            test_centroid = compute_fn(test_outputs, dim)
            distance_list.append(
                euclidean_distance(train_centroid, test_centroid)
            )
    #! FIXME a quick way to fix the nan/inf issue
    avg_distance = np.array(np.isfinite(distance_list))
    avg_distance = np.mean(avg_distance)
    return avg_distance


def compute_centroid_for_each_setup(setup_path, dataset, data_dir, idx_file, model_num_per_repeat,
                                    out_dir, setup_tag, compute_tag, repeat_num):
    model_dir = os.path.join(setup_path, "models")
    config_file = os.path.join(setup_path, "config.ini")
    train_config, model_config = train.load_config(config_file)
    epoch_num = train_config["epoch_num"]
    out_dir = os.path.join(out_dir, "last_layer")
    if os.path.isdir(out_dir):
        print(f"[{setup_tag}] Existed!")
    else:
        os.mkdir(out_dir)
        # load dataset
        train_set, test_set = train.load_complete_data(dataset, data_dir)
        idx_data = utils.read_npy(idx_file)
        test_loader = train.prepare_test_set(
            dataset, test_set, idx_data["test_idx_set"], int(train_config["test_batch_size"]))

        model_path_array = estimate.get_model_paths(
            model_dir, model_num_per_repeat, repeat_num)
        repeat_id = 0
        for model_paths in model_path_array:
            repeat_id += 1
            get_last_layer_for_each_setup(
                repeat_id, model_paths, out_dir, model_config, test_loader, train_set,
                idx_data["train_idx_sets"],
                int(train_config["test_batch_size"]),
                dataset
            )
        print(f"[{setup_tag}] Finished!")
        assert (len(os.listdir(out_dir)) !=
                0), "The last_layer folder is empty."
    avg_distance = compute_train_test_centroid(out_dir, compute_tag)
    assert (not np.isinf(avg_distance)), "avg_distance centroid is inf"
    assert (not np.isnan(avg_distance)), "avg_distance centroid is nan"
    return avg_distance, epoch_num


def one_setup_main(args):
    # process arguments
    tmp = args.model_dirs.split(",")
    model_dirs = []
    for model_dir in tmp:
        if os.path.isdir(model_dir):
            model_dirs.append(model_dir)
    if len(model_dirs) == 0:
        raise Exception(f"Error: No valid model folder ... {model_dir}")
    datasets = args.datasets.split(",")
    epoch_nums = [int(num) for num in args.epoch_nums.split(",")]
    schedulers = args.schedulers.split(",")
    losses = args.losses.split(",")
    archs = args.archs.split(",")
    widths = [int(width) for width in args.widths.split(",")]
    ntrains = args.ntrains.split(",")
    if not os.path.isdir(args.data_dir):
        raise Exception(f"Error: Cannot find the data dir ... {args.data_dir}")
    if not os.path.exists(args.data_idx_path):
        raise Exception(
            f"Error: The data index file is not valid ... {args.data_idx_path}")

    list_of_setups = product(epoch_nums, schedulers,
                             losses, datasets, archs, widths, ntrains)
    for setup in list_of_setups:
        epoch_num = setup[0]
        scheduler = setup[1]
        loss = setup[2]
        dataset = setup[3]
        arch = setup[4]
        width = setup[5]
        ntrain = setup[6]
        tag = False
        cnt = 0
        for model_dir in model_dirs:
            cnt += 1
            setup_path = os.path.join(
                model_dir,
                f"epoch_{epoch_num}/{scheduler}-{loss}/{dataset}-{arch}-w_{width}-ntrain_{ntrain}-N_{args.repeat_num}"
            )
            if os.path.isdir(setup_path) and arch == "alexnetwol":
                setup_path = f"{setup_path}-less_lr"
            if cnt == 1:
                if os.path.isdir(setup_path):
                    out_dir = deepcopy(setup_path)
                else:
                    raise Exception(
                        f"Error: Cannot find the folder ... {out_dir}")
            tmp = os.path.join(setup_path, "models")
            if os.path.isdir(tmp):
                tag = True
                break
        if not tag:
            print(
                f"Error: Cannot find the setup folder for *epoch_{epoch_num}/{scheduler}-{loss}/{dataset}-{arch}-w_{width}-ntrain_{ntrain}-N_{args.repeat_num}*")
            continue
        distance_list = []
        for compute_tag in ComputeTags:
            distance, config_epoch_num = compute_centroid_for_each_setup(
                setup_path, dataset, args.data_dir, args.data_idx_path,
                args.model_num_per_repeat, out_dir,
                f"epoch_{epoch_num}/{scheduler}-{loss}/{dataset}-{arch}-w_{width}-ntrain_{ntrain}-N_{args.repeat_num}",
                compute_tag,
                args.repeat_num
            )
            assert (not np.isnan(distance)), "centroid distance is nan"
            assert (not np.isinf(distance)), "centroid distance is inf"
            assert (epoch_num == config_epoch_num), "different epoch num in config!"
            distance_list.append(str(distance))
        utils.write_txt(
            os.path.join(out_dir, "centroid_distance.txt"),
            [",".join(ComputeTags) + "\n", ",".join(distance_list) + "\n"]
        )


def run_all(args):
    models_dirs = args.models_dirs
    data_dir = args.data_dir

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    list_of_setups = product(EpochNums, SchedulerList, LossDict.keys(), WidthDict.keys(), TrainSizeList)
    for setup in list_of_setups:
        epoch_num = setup[0]
        scheduler = setup[1]
        loss = setup[2]
        arch = setup[3]
        ntrain = setup[4]
        for dataset in LossDict[loss]:
            for width in WidthDict[arch]:
                if dataset == "mnist" and not arch == "mlp_1hidden":
                    continue
                tag = False
                if dataset == "mnist":
                    model_num_per_repeat = 60
                else:
                    model_num_per_repeat = ModelNumPerRepeat[ntrain]
                setup_path = os.path.join(
                    models_dirs,
                    f"epoch_{epoch_num}/{scheduler}-{loss}/{dataset}-{arch}-w_{width}-ntrain_{ntrain}-N_{RepeatNum}"
                )
                if arch == "alexnetwol":
                    setup_path = f"{setup_path}-less_lr"
                out_dir = setup_path
                tmp = os.path.join(setup_path, "models")
                if os.path.isdir(tmp):
                    tag = True
                if not tag:
                    print(
                        f"Error: Cannot find the setup folder for *epoch_{epoch_num}/{scheduler}-{loss}/{dataset}-{arch}-w_{width}-ntrain_{ntrain}-N_{RepeatNum}*")
                    continue
                distance_list = []
                train_size = TRAIN_SIZE_DICT[ntrain]
                data_idx_path = os.path.join(data_dir,
                                            f'prepare_data-data_{dataset}-ntrial_{RepeatNum}-train_size_{train_size}-test_size_10000-strategy_disjoint_split-seed_0.npz')
                for compute_tag in ComputeTags:
                    distance, config_epoch_num = compute_centroid_for_each_setup(
                        setup_path, dataset, data_dir, data_idx_path,
                        model_num_per_repeat, out_dir,
                        f"epoch_{epoch_num}/{scheduler}-{loss}/{dataset}-{arch}-w_{width}-ntrain_{ntrain}-N_{RepeatNum}",
                        compute_tag,
                        RepeatNum
                    )
                    assert (not np.isnan(distance)), "centroid distance is nan"
                    assert (not np.isinf(distance)), "centroid distance is inf"
                    assert (config_epoch_num == epoch_num), "different epoch_num in config!"
                    distance_list.append(str(distance))
                utils.write_txt(
                    os.path.join(out_dir, "centroid_distance.txt"),
                    [",".join(ComputeTags) + "\n", ",".join(distance_list) + "\n"]
                )


parser = argparse.ArgumentParser(description="Compute Centroid")
subparsers = parser.add_subparsers(
    help="Either run for one setup or run for all setups")

all_parser = subparsers.add_parser("all")
all_parser.add_argument(
    "--data_dir", dest="data_dir", type=str, required=True, help="Run all the setups from the data_dir."
)
all_parser.add_argument(
    "--models_dir", dest="models_dirs", type=str, required=True, help="Run all the setups for the models in models_dir."
)
all_parser.add_argument(
    "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
)
all_parser.set_defaults(func=run_all)

one_parser = subparsers.add_parser("one_setup")
one_parser.add_argument(
    "--model_dirs", "-ms", dest="model_dirs", type=str, required=True, help="Path to model folders"
)
one_parser.add_argument(
    "--data_dir", "-dd", dest="data_dir", type=str, required=True, help="Path to the data folder"
)
one_parser.add_argument(
    "--data_idx_path", "-i", dest="data_idx_path", type=str, required=True, help="Path to the data index file for training"
)
one_parser.add_argument(
    "--datasets", "-d", dest="datasets", type=str, required=True, help="Datasets"
)
one_parser.add_argument(
    "--archs", "-a", dest="archs", type=str, required=True, help="Model architectures"
)
one_parser.add_argument(
    "--widths", "-w", dest="widths", type=str, required=True, help="Model width"
)
one_parser.add_argument(
    "--ntrains", "-n", dest="ntrains", type=str, required=True, help="Number of training samples used for training"
)
one_parser.add_argument(
    "--repeat_num", "-r", dest="repeat_num", type=int, required=False, default=3, help="Number of repeats"
)
one_parser.add_argument(
    "--epoch_nums", "-e", dest="epoch_nums", type=str, required=True, help="Number of epochs for training"
)
one_parser.add_argument(
    "--schedulers", "-s", dest="schedulers", type=str, required=True, help="Whether to use scheduler for training"
)
one_parser.add_argument(
    "--losses", "-l", dest="losses", type=str, required=True, help="Which loss to use for training"
)
one_parser.add_argument(
    "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
)
one_parser.add_argument(
    "--model_num_per_repeat", "-mn", dest="model_num_per_repeat", type=int, required=True, help="Number of splits created for training"
)
one_parser.set_defaults(func=one_setup_main)


if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
