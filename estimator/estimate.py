import os
import sys
import pdb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import loader.utils as utils
import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
from trainer import models
from trainer import train
import torch.nn.functional as  function
import time, math


RandomSeedForData = "0"
TrainSize = {
    "mnist": 60*1000,
    "cifar10": 50*1000,
    "cifar100": 50*1000,
}

# Define loss
criterion_list = {
    "mse": nn.MSELoss(reduction='mean').cuda(),
    "ce": nn.CrossEntropyLoss(reduction="mean").cuda()
}
raw_criterion_list = {
    "mse": nn.MSELoss(reduction="none").cuda(),
    "ce": nn.CrossEntropyLoss(reduction="none").cuda()
}

nll_loss = nn.NLLLoss(size_average=False)


def calc_kl_div(p,q):
    return (p * (p / q).log()).sum()

def compute_log_output(model, testloader, class_num, test_size):
    model.eval()
    cnt = 0
    #logprobs= torch.Tensor(test_size, class_num).zero_().cuda()
    logprobs = torch.Tensor(test_size, class_num).zero_().type(torch.DoubleTensor).cuda()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs = inputs.cuda()
            input_num = labels.size(0)
            labels = labels.long().cuda()

            outputs = model(inputs)

            outputs = outputs.type(torch.DoubleTensor).cuda()

            outputs = function.softmax(outputs, dim=1)
            logprobs[cnt:(cnt+input_num), :] = outputs.log()
            cnt += input_num
    return logprobs

def compute_normalized_kl(logprobs_avg, class_num, test_size):
    #outputs = torch.Tensor(test_size, class_num).zero_().cuda()
    outputs = torch.Tensor(test_size, class_num).zero_().type(torch.DoubleTensor).cuda()
    for sample_idx in range(test_size):
        prob_sum = 0.0
        for class_idx in range(class_num):
            outputs[sample_idx, class_idx] = torch.exp(logprobs_avg[sample_idx, class_idx])
            prob_sum += outputs[sample_idx, class_idx]
        outputs[sample_idx, :] = outputs[sample_idx, :] / float(prob_sum)
    return outputs

def log_mse(log_path, prefix, repeat_num, model_num, loss, acc, variance, bias_square, unbiased_variance, unbiased_bias_square):
    log_str = "%s: %s\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f" % (
        str(prefix, time.time()), repeat_num, model_num, loss, acc, variance, bias_square, unbiased_variance, unbiased_bias_square
    )
    print(log_str)
    utils.write_txt(log_path, [log_str+"\n"])

def log_ce(log_path, repeat_num, model_num, loss, acc, variance, bias_square):
    log_str = "CE: %s\t%d\t%d\t%f\t%f\t%f\t%f" % (
        str(time.time()), repeat_num, model_num, loss, acc, variance, bias_square
    )
    print(log_str)
    utils.write_txt(log_path, [log_str+"\n"])


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

def compute_test_avg_log_output(data_size, data_loader, model_config, model_path_array_dict):
    # for cross entropy loss
    # compute log-output average
    # logprobs_sum = torch.Tensor(test_size, int(model_config["out_dim"])).zero_().cuda()
    logprobs_sum = torch.Tensor(data_size, int(model_config["out_dim"])).zero_().type(torch.DoubleTensor).cuda()
    model_cnt = 0
    for model_paths_dict in model_path_array_dict:
        for model_id in model_paths_dict:
            try:
                model_cnt += 1
                model_path = model_paths_dict[model_id]
                model = models.MODELS[model_config["model_tag"]](model_config).cuda()
                model.load_state_dict(torch.load(model_path))
                logprobs_sum += compute_log_output(model, data_loader, int(model_config["out_dim"]), data_size)
            except Exception as e:
                print("Error:", e)
                print("[compute_test_avg_log_output] File is corrupted {}!".format(model_path))
                with open("corrupted_file_model.txt", "a+") as f:
                    f.write("{}\n".format(model_path))
                continue
    logprobs_avg = logprobs_sum / float(model_cnt)
    normalized_avg = compute_normalized_kl(logprobs_avg, int(model_config["out_dim"]), data_size)
    return normalized_avg

def compute_train_avg_log_output(data_size, data_loader_array, data_idx_array, model_config, model_path_array_dict, model_num_per_repeat):
    logprobs_sum = torch.Tensor(data_size, int(model_config["out_dim"])).zero_().type(torch.DoubleTensor).cuda()
    cnt_sum = torch.Tensor(data_size).zero_().type(torch.DoubleTensor).cuda()
    for repeat_id in range(len(model_path_array_dict)):
        model_paths_dict = model_path_array_dict[repeat_id]
        data_loaders = data_loader_array[repeat_id]
        for model_id in model_paths_dict:
            model_index_for_current_repeat = model_id - model_num_per_repeat*repeat_id - 1 # the data_laoder and data_idx_array are for each repeat, so we need to do this subtraction
            model_path = model_paths_dict[model_id]
#            print("Length of data_loaders: {} and index: {}".format(len(data_loaders), model_id-1))
            data_loader = data_loaders[model_index_for_current_repeat]
            data_idx_list = data_idx_array[repeat_id][model_index_for_current_repeat]
            model = models.MODELS[model_config["model_tag"]](model_config).cuda()
            try:
                model.load_state_dict(torch.load(model_path))
            except:
                print("[compute_train_avg_log_output] File is corrupted {}!".format(model_path))
                with open("corrupted_file_model.txt", "a+") as f:
                    f.write("{}\n".format(model_path))
                continue
            logprobs_sum[data_idx_list, :] += compute_log_output(model, data_loader, int(model_config["out_dim"]), len(data_idx_list))
            cnt_sum[data_idx_list] += 1
    logprobs_avg = torch.div(logprobs_sum.transpose(0, 1), cnt_sum).transpose(0,1)
    normalized_avg = compute_normalized_kl(logprobs_avg, int(model_config["out_dim"]), data_size)
    return normalized_avg


def compute_bias_variance_mse(model, data_loader, class_num, output_collection, norm_square_output_collection, acc_collection, model_num):
    # Set to be evaluation model
    model.eval()

    correct_num = 0
    bias_square = 0
    cnt = 0
    loss = 0
    variance = 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            # Compute outputs
            inputs = inputs.cuda()
            # Prepare labels
            labels = labels.long().cuda()
            input_num = labels.size(0)
            labels_onehot = torch.FloatTensor(input_num, class_num).cuda()
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.view(-1, 1).long(), 1)

            # Compute loss which is averaged over each element (note: not just over each x)
            outputs = model(inputs)
            tmp = utils.criterion_list["mse"](outputs, labels_onehot)
            # Compute accuracy
            _, predictions = outputs.max(1)
            correct_num += predictions.eq(labels).sum().item()

            # Sum_{x,y}[(f(x) - y)^2]
            loss += tmp.item() * outputs.numel() # numel() give you the number of element
            # Sum[f(x)]
            output_collection[cnt: (cnt + input_num), :] += outputs
            # E_{D}Sum_{x,y}[(f_bar(x) - y)^2]
            bias_square += (output_collection[cnt: (cnt + input_num), :]/model_num - labels_onehot).norm() ** 2.0
            # Sum[f^2(x)]
            norm_square_output_collection[cnt: (cnt + input_num)] += outputs.norm(dim=1) ** 2.0
            # Var(f(x)) = E[(f(x) - f_bar(x))^2] = E[f^2(x)] - (E[f(x)])^2
            # E_{D}Sum_{x}[(f(x) - f_bar(x))^2] = Sum[f^2(x)] - Sum[(E[f(x)])^2]
            variance += norm_square_output_collection[cnt: (cnt + input_num)].sum()/model_num - (output_collection[cnt: (cnt + input_num), :]/model_num).norm() ** 2.0
            cnt += input_num
    # percentage
    acc_collection = acc_collection +  correct_num / float(cnt)
    return bias_square/cnt, variance/cnt, output_collection, norm_square_output_collection, loss/cnt, acc_collection

def compute_bias_variance_mse_train(model, data_loader, idx_list, class_num, output_collection, norm_square_output_collection, loss_per_element,  acc_collection):
    model.eval()

    correct_num = 0
    cnt = 0
    loss = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            # Compute outputs
            inputs = inputs.cuda()
            # Prepare labels
            labels = labels.long().cuda()
            input_num = labels.size(0)
            labels_onehot = torch.FloatTensor(input_num, class_num).cuda()
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.view(-1, 1).long(), 1)

            # Compute loss which is averaged over each element (note: not just over each x)
            outputs = model(inputs)
            tmp = utils.criterion_list["mse"](outputs, labels_onehot)
            # Compute accuracy
            _, predictions = outputs.max(1)
            correct_num += predictions.eq(labels).sum().item()

            # Sum_{x,y}[(f(x) - y)^2]
            loss += tmp.item() * outputs.numel()
            tmp_raw = utils.raw_criterion_list["mse"](outputs, labels_onehot)
            mean_loss_raw = torch.sum(tmp_raw, dim=1) # mse loss of x, y (vectors) is l = ||(x-y)||^2
            loss_per_element[idx_list[cnt: (cnt + input_num)]] += mean_loss_raw # add the mean loss to every position corresponding to the given elements via idx_list
            # Sum[f(x)]
            output_collection[idx_list[cnt: (cnt + input_num)], :] += outputs
            # Sum[f^2(x)]
            norm_square_output_collection[idx_list[cnt: (cnt + input_num)]] += outputs.norm(dim=1) ** 2.0
            cnt += input_num
    acc_collection = acc_collection + correct_num / float(cnt)
    return output_collection, norm_square_output_collection, loss_per_element, loss/cnt, acc_collection

def estimate_mse_train(model_config, data_size, data_loader, model_paths_dict, repeat_id, model_num_per_repeat, data_idx_array, output_collection, norm_square_output_collection, cnt_element, loss_per_element,log_path):
    class_num = int(model_config["out_dim"])
    loss_collection = 0
    acc_collection = 0
    detailed_losses = []
    total_data_size = 0
    model_num = 0 
    for model_id in model_paths_dict:
        model_path = model_paths_dict[model_id]
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print("[estimate_mse_train] File is corrupted {}!".format(model_path))
            with open("corrupted_file_model.txt", "a+") as f:
                f.write("{}\n".format(model_path))
            continue
        model_num += 1
        model_index_for_current_repeat = model_id - model_num_per_repeat*repeat_id - 1 # the data_laoder and data_idx_array are for each repeat, so we need to do this subtraction
        data_loader_for_model = data_loader[model_index_for_current_repeat]
        data_idx_for_model = data_idx_array[model_index_for_current_repeat]
        output_collection, norm_square_output_collection, loss_per_element, loss, acc_collection = compute_bias_variance_mse_train(model, data_loader_for_model, data_idx_for_model, class_num, output_collection, norm_square_output_collection, loss_per_element, acc_collection)
        cnt_element[data_idx_for_model] += 1
        loss_collection += loss
        detailed_losses.append(loss)
        total_data_size += len(data_loader[model_index_for_current_repeat])
    return loss_collection/model_num, acc_collection/model_num, detailed_losses, total_data_size, output_collection, norm_square_output_collection, cnt_element, loss_per_element

def estimate_mse_test(model_config, data_size, data_loader, model_paths_dict, repeat_id, model_num_per_repeat, log_path):
    class_num = int(model_config["out_dim"])
    output_collection = torch.Tensor(data_size, class_num).zero_().cuda()
    norm_square_output_collection = torch.Tensor(data_size).zero_().cuda()
    loss_collection = 0
    acc_collection = 0
    detailed_losses = []
    model_num = 0
    for model_id in model_paths_dict:
        model_path = model_paths_dict[model_id]
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print("[estimate_mse_test] File is corrupted {}!".format(model_path))
            with open("corrupted_file_model.txt", "a+") as f:
                f.write("{}\n".format(model_path))
            continue
        model_num += 1
        bias_square, variance, output_collection, norm_square_output_collection, loss, acc_collection = compute_bias_variance_mse(model, data_loader, class_num, output_collection, norm_square_output_collection, acc_collection, model_num)
        loss_collection += loss
        detailed_losses.append(loss)
    unbiased_variance = variance * model_num / (model_num - 1.0)
    unbiased_bias_square = loss_collection/model_num - unbiased_variance
    return loss_collection/model_num, acc_collection/model_num, unbiased_variance, unbiased_bias_square, detailed_losses


def compute_bias_variance_ce(model, data_loader, normalized_avg):
    model.eval()
    cnt = 0
    bias2 = 0
    variance = 0
    test_loss = 0
    correct_num = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            input_num = labels.size(0)

            outputs = model(inputs)

            outputs = outputs.type(torch.DoubleTensor).cuda()
            # outputs = outputs.cuda()

            batch_loss = utils.criterion_list["ce"](outputs, labels)
            test_loss += batch_loss.item() * input_num
            _, preds = outputs.max(1)
            correct_num += preds.eq(labels).sum().item()

            outputs = function.softmax(outputs, dim=1)
            bias2 += nll_loss(normalized_avg[cnt:cnt+input_num, :].log(), labels)
            for sample_idx in range(input_num):
                sample_var = calc_kl_div(normalized_avg[cnt+sample_idx], outputs[sample_idx])
                assert(sample_var > - 0.0001)
                variance += sample_var
            cnt += input_num
    return test_loss/cnt, correct_num/float(cnt), bias2/cnt, variance/cnt

def compute_bias_variance_ce_train(model, data_loader, normalized_avg, idx_list):
    model.eval()

    test_loss = 0
    correct_num = 0
    bias2 = 0
    cnt = 0
    variance = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            input_num = labels.size(0)

            outputs = model(inputs)

            outputs = outputs.type(torch.DoubleTensor).cuda()

            batch_loss = criterion_list["ce"](outputs, labels)
            test_loss += batch_loss.item() * input_num
            _, preds = outputs.max(1)
            correct_num += preds.eq(labels).sum().item()

            outputs = function.softmax(outputs, dim=1)

            bias2 += nll_loss(normalized_avg[idx_list[cnt:cnt+input_num], :].log(), labels)
            for sample_idx in range(input_num):
                sample_var = calc_kl_div(normalized_avg[idx_list[cnt+sample_idx]], outputs[sample_idx])
                assert(sample_var > - 0.0001)
                variance += sample_var

            cnt += input_num
    return test_loss/cnt, correct_num/float(cnt), bias2/cnt, variance/cnt


def estimate_ce(model_config, data_loader, model_paths_dict, normalized_avg, repeat_id, model_num_per_repeat, data_idx_array):
    tag = False
    if data_idx_array is not None:
        tag = True

    model_num = 0
    variance_sum = 0.0
    loss_sum = 0.0
    acc_sum = 0.0
    bias2_sum = 0.0
    detailed_losses = []
    for model_id in model_paths_dict:
        model_path = model_paths_dict[model_id]
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print("[estimate_mse_test] File is corrupted {}!".format(model_path))
            with open("corrupted_file_model.txt", "a+") as f:
                f.write("{}\n".format(model_path))
            continue
        model_num += 1
        if tag:
            model_index_for_current_repeat = model_id - model_num_per_repeat*repeat_id - 1 # the data_laoder and data_idx_array are for each repeat, so we need to do this subtraction
            loss, acc, bias2, variance = compute_bias_variance_ce_train(model, data_loader[model_index_for_current_repeat], normalized_avg, data_idx_array[model_index_for_current_repeat])
        else:
            loss, acc, bias2, variance = compute_bias_variance_ce(model, data_loader, normalized_avg)
        variance_sum += variance
        loss_sum += loss
        acc_sum += acc
        bias2_sum += bias2
        detailed_losses.append(loss)
    return loss_sum/model_num, acc_sum/model_num, variance_sum/model_num, bias2_sum/model_num, detailed_losses

def compute_statistics(data_size, data_loader, train_config, model_config, model_paths_dict, normalized_avg, repeat_id, model_num_per_repeat, data_idx_array):
    # compute the bias2 & variance
    # if train_config["loss"] == "mse":
    #     loss, acc, var, bias2, detailed_losses = estimate_mse(model_config, data_size, data_loader, model_paths_dict, repeat_id, model_num_per_repeat, data_idx_array)
    if train_config["loss"] == "ce":
        loss, acc, var, bias2, detailed_losses = estimate_ce(model_config, data_loader, model_paths_dict, normalized_avg, repeat_id, model_num_per_repeat, data_idx_array)
    else:
        raise Exception("Error: The estimation function is not defined!")
    return loss, acc, var.cpu().numpy(), bias2.cpu().numpy(), detailed_losses

def prepare_train_loader(data_tag, train_set, train_idx_data, model_num_per_repeat, train_config):
    train_loader_array = []
    train_idx_array = []
    tmp = []
    tmp2 = []
    for idx in range(len(train_idx_data)):
        if idx != 0 and idx % model_num_per_repeat == 0:
            train_loader_array.append(tmp)
            train_idx_array.append(tmp2)
            tmp = []
            tmp2 = []
        tmp.append(
            train.prepare_test_set(data_tag, deepcopy(train_set), train_idx_data[idx], int(train_config["test_batch_size"]))
        )
        tmp2.append(train_idx_data[idx])
    train_loader_array.append(tmp)
    train_idx_array.append(tmp2)
    return train_loader_array, train_idx_array

def get_stats(stats_dict):
    avg_test_loss = np.mean(np.asarray(stats_dict["test_loss"]))
    avg_test_acc = np.mean(np.asarray(stats_dict["test_acc"]))
    avg_test_var = np.mean(np.asarray(stats_dict["test_var"]))
    avg_test_bias2 = np.mean(np.asarray(stats_dict["test_bias2"]))

    avg_train_loss = np.mean(np.asarray(stats_dict["train_loss"]))
    avg_train_acc = np.mean(np.asarray(stats_dict["train_acc"]))
    avg_train_var = np.mean(np.asarray(stats_dict["train_var"]))
    avg_train_bias2 = np.mean(np.asarray(stats_dict["train_bias2"]))

    return avg_test_loss, avg_test_acc, avg_test_var, avg_test_bias2, avg_train_loss, avg_train_acc, avg_train_var, avg_train_bias2

def get_model_paths(model_dir, model_num_per_repeat, num_repeats):
    file_list = os.listdir(model_dir)
    file_dict = {}
    for file_name in file_list:
        if file_name[:11] == "final_model":
            file_id = int(file_name.split('.')[0].split("_")[-1])
            file_dict[file_id] = os.path.join(model_dir, file_name)
    file_array = []
    # Teo: for missing models we cannot get the number of repeats by dividing the size
    # of the dictionary with the number of models per repeat
    #round_num = len(file_dict) // model_num_per_repeat
    if len(file_dict) * model_num_per_repeat != num_repeats:
        print("WARNING! Missing models for {}".format(model_dir))

    for round_id in range(num_repeats):
        tmp = []
        for model_id in range(model_num_per_repeat):
            file_id = model_id + round_id * model_num_per_repeat + 1
            if file_id in file_dict:
                tmp.append(file_dict[file_id])
        file_array.append(tmp)
    assert (len(file_array) != 0), f"No files found in {model_dir}"
    return file_array

def get_model_paths_dicts(model_dir, model_num_per_repeat, N=3):
    file_list = os.listdir(model_dir)
    file_dict = {}
    for file_name in file_list:
        if file_name[:11] == "final_model":
            file_id = int(file_name.split('.')[0].split("_")[-1])
            file_dict[file_id] = os.path.join(model_dir, file_name)
    dict_array = []
    round_num = N
    for round_id in range(round_num):
        tmp = {}
        for model_id in range(model_num_per_repeat):
            file_id = model_id + round_id * model_num_per_repeat + 1
            if file_id not in file_dict:
                print('Warning: File_id {} missing'.format(file_id))
            else:
                tmp[file_id] = file_dict[file_id]
        dict_array.append(tmp)
    return dict_array

def log_args(args, out_dir):
    out_path = os.path.join(out_dir, "estimate_args.txt")
    in_str = []
    for key in sorted(list(args.keys())):
        in_str.append("%s: %s\n" % (str(key), str(args[key])))
    utils.write_txt(out_path, in_str)


def analyze_each_setup(data_dir, setup_dir_path, FORCE_STAT_REFRESH = False, failed_stat_log_path = None):
    print("Starting %s" % setup_dir_path)
    config_path = os.path.join(setup_dir_path, "config.ini")

    # load config
    if not os.path.exists(config_path):
        print("Error: Configuration file does not exist ... %s" % config_path)
        return False, None
    train_config, model_config = train.load_config(config_path)

    out_path = os.path.join(setup_dir_path, "stats.pkl")
    log_path = os.path.join(setup_dir_path, "stats.log")

    # Sanity test for stats
    print("Performing Sanity Test...")
    valid_stat = True
    if os.path.exists(out_path):
        try:
            stats = utils.read_pkl(out_path)
            lst = get_stats(stats)
            for i in range(len(lst)):
                val = lst[i]
                if math.isnan(val) or (val <= 0)  :
                    print("Error: Stat value at index {} is {}!".format(i,val))
                    valid_stat = False
                    break
            if not valid_stat:
                print("Forcing Stat Refresh!")
                FORCE_STAT_REFRESH = True
        except Exception as e:
            print("Error: Can't get the stats!\n", e)
            valid_stat = False
            FORCE_STAT_REFRESH = True

    if not valid_stat:
        print("Sanity test failed...")
        if failed_stat_log_path:
            with open(failed_stat_log_path, "a+") as f:
                f.writelines([setup_dir_path + "\n"]) # write to log if path provided
                f.flush()

    if valid_stat and not FORCE_STAT_REFRESH and os.path.exists(out_path):
        stats = utils.read_pkl(out_path)
        avg_test_loss, avg_test_acc, avg_test_var, avg_test_bias2, avg_train_loss, avg_train_acc, avg_train_var, avg_train_bias2 = get_stats(stats)
        print("Loaded computed stats ...")
        return True, [avg_train_acc, avg_test_acc, avg_train_loss, \
                      avg_test_loss, avg_train_var, avg_test_var, \
                      avg_train_bias2, avg_test_bias2, \
                      avg_train_acc - avg_test_acc, avg_test_loss - avg_train_loss, \
                      None]

    # load idx data
    idx_file, data_tag, train_subset_size, split_num = get_idx_file(data_dir, setup_dir_path)
    idx_data = utils.read_npy(idx_file)
    # load data
    train_set, test_set = train.load_complete_data(data_tag, data_dir)
    train_size = train_set.data.shape[0]
    test_size = len(idx_data["test_idx_set"])

    model_dir = os.path.join(setup_dir_path, "models")
    model_num_per_repeat = int(TrainSize[data_tag]/train_subset_size)
    total_model_num = model_num_per_repeat * split_num
    if (not os.path.isdir(model_dir)):
        print("Error: Cannot find the folder ... %s" % model_dir)
        return False, None
    if len(os.listdir(model_dir)) < total_model_num:
        print("Warning: The number of models are not sufficient ... %d/%d ... %s" % (
            len(os.listdir(model_dir)), total_model_num, model_dir
        ))
    #    return False, None
    model_path_array_dict = get_model_paths_dicts(model_dir, model_num_per_repeat, split_num)
    print("Debug: Using the model_path_array_dictionary - ", model_path_array_dict)
    if len(model_path_array_dict) != split_num:
        return False, None

    # prepare test loader
    test_loader = train.prepare_test_set(data_tag, test_set, idx_data["test_idx_set"], int(train_config["test_batch_size"]))
    train_loader_array, train_idx_array = prepare_train_loader(data_tag, train_set, idx_data["train_idx_sets"], model_num_per_repeat, train_config)
    if len(train_loader_array) != split_num:
        return False, None

    test_loss_sum = []
    test_acc_sum = []
    test_var_sum = []
    test_bias2_sum = []

    train_loss_sum = []
    train_acc_sum = []
    train_var_sum = []
    train_bias2_sum = []
    detailed_losses = {}

    # Computing the stats if loss is MSE
    if train_config["loss"] == "mse": # todo: hack
        # start with test set
        for repeat_id in range(split_num):
            print("Debug: Start repeat %d" % repeat_id)
            model_paths_dict = model_path_array_dict[repeat_id]
            test_loss, test_acc, test_var, test_bias2, test_detail_losses =  estimate_mse_test(model_config, test_size, test_loader, model_paths_dict, repeat_id, model_num_per_repeat, log_path)
            detailed_losses[repeat_id] = {"test": test_detail_losses}
            test_loss_sum.append(float(test_loss))
            test_acc_sum.append(float(test_acc))
            test_var_sum.append(float(test_var))
            test_bias2_sum.append(float(test_bias2))
            print("Debug: Finish computing stats on testing samples")

        print("Test var: {} \nTest Bias2: {}\nTest Loss: {}".format(test_var_sum, test_bias2_sum, test_loss_sum))
        # start with training set
        # print("Starting computation for the training samples for the MSE model...")
        class_num = int(model_config["out_dim"])
        # we will go through all the model_ids, and keep udpating the following data structures, finally we will compute the stats
        output_collection = torch.Tensor(train_size, class_num).zero_().cuda()
        norm_square_output_collection = torch.Tensor(train_size).zero_().cuda()
        cnt_element = torch.Tensor(train_size).zero_().type(torch.DoubleTensor).cuda() # this will store the number of times we update/add to the output_collection and norm_square_output_collection 
        loss_per_element = torch.Tensor(train_size).zero_().type(torch.DoubleTensor).cuda() # this will store the total loss of each element by adding to the loss to corresponding location every time we process that element through a model
        total_data_size_lst = []

        for repeat_id in range(split_num):
            print("Debug: Start repeat %d" % repeat_id)
            model_paths_dict = model_path_array_dict[repeat_id]
            train_loaders = train_loader_array[repeat_id]
            data_idx_array_repeat = train_idx_array[repeat_id]
            train_loss, train_acc, train_detail_losses, total_data_size, output_collection, norm_square_output_collection, cnt_element, loss_per_element = estimate_mse_train(model_config, train_size, train_loaders, model_paths_dict, repeat_id, model_num_per_repeat, data_idx_array_repeat, output_collection, norm_square_output_collection, cnt_element, loss_per_element, log_path)
            detailed_losses[repeat_id] = {} if repeat_id not in detailed_losses else detailed_losses[repeat_id]
            detailed_losses[repeat_id]["train"] = train_detail_losses
#            train_loss_sum.append(float(train_loss))
            train_acc_sum.append(float(train_acc))
            total_data_size_lst.append(total_data_size)

        # now we can compute the variance using norm_square_output_collection, output_collection, and cnt_element
        # To compute the variance, we first have to restrict out losses, norm_square_output_collection, output_collection to the positions where [cnt_element >= 2]
        cnt_filter = (cnt_element >= 2)
        total_data_size = sum(cnt_filter)
        if total_data_size  == 0:
            raise Exception("Not enough data for computation of training variance for the model.")
        print("Debug: Total Data Size after filtering: ", total_data_size) 

        filter_loss_per_element = loss_per_element[cnt_filter]
        filter_norm_square_output_collection = norm_square_output_collection[cnt_filter]
        filter_output_collection = output_collection[cnt_filter]
        cnt_filter_values = cnt_element[cnt_filter]
        avg_filter_loss_per_element = filter_loss_per_element/cnt_filter_values
        avg_filter_norm_square_output_collection  = filter_norm_square_output_collection/cnt_filter_values
        avg_filter_output_collection  = torch.div(filter_output_collection, cnt_filter_values[:, None])

        # to compute the number of elements that are involved in the computation of variance/bias, we find the number of elements that had [cnt_element >2]
        total_data_size = sum(cnt_filter)
        train_var = (avg_filter_norm_square_output_collection.sum() - (avg_filter_output_collection.norm() ** 2))/total_data_size
        total_loss_filter = sum(avg_filter_loss_per_element)/total_data_size
        train_bias2 = total_loss_filter - train_var
        # now we append these single values in the list for consistency with other code
        train_loss_sum.append(float(total_loss_filter))
        train_var_sum.append(float(train_var))
        train_bias2_sum.append(float(train_bias2))
        print("Debug: Finish computing stats on training samples")
        print("Train var: {} \nTrain Bias2: {}\nTrain Loss(filtered): {}\nTrain Loss(all): {}".format(train_var_sum, train_bias2_sum, train_loss_sum, train_loss))

    if train_config["loss"] == "ce":
        print("Debug: Computing the normalized averages for the CE model...")
        test_normalized_avg = compute_test_avg_log_output(test_size, test_loader, model_config, model_path_array_dict) #, model_num_per_repeat)
        train_normalized_avg = compute_train_avg_log_output(train_size, train_loader_array, train_idx_array, model_config, model_path_array_dict, model_num_per_repeat)
        print("Debug: Done computing the normalized averages...")
        for repeat_id in range(split_num):
            print("Debug: Start repeat %d" % repeat_id)
            model_paths_dict = model_path_array_dict[repeat_id]
            train_loaders = train_loader_array[repeat_id]

            # start with test set
            test_loss, test_acc, test_var, test_bias2, test_detail_losses = compute_statistics(test_size, test_loader, train_config, model_config, model_paths_dict, test_normalized_avg, repeat_id, model_num_per_repeat, None)
            detailed_losses[repeat_id] = {"test": test_detail_losses}
            test_loss_sum.append(float(test_loss))
            test_acc_sum.append(float(test_acc))
            test_var_sum.append(float(test_var))
            test_bias2_sum.append(float(test_bias2))
            print("Debug: Finish computing stats on testing samples")
            print("Test var: {} \nTest Bias2: {}\nTest Loss: {}".format(test_var_sum, test_bias2_sum, test_loss_sum))

            # start with train set
            train_loss, train_acc, train_var, train_bias2, train_detail_losses = compute_statistics(train_size, train_loaders, train_config, model_config, model_paths_dict, train_normalized_avg, repeat_id, model_num_per_repeat, train_idx_array[repeat_id])
            detailed_losses[repeat_id]["train"] = train_detail_losses
            train_loss_sum.append(float(train_loss))
            train_acc_sum.append(float(train_acc))
            train_var_sum.append(float(train_var))
            train_bias2_sum.append(float(train_bias2))
            print("Debug: Finish computing stats on training samples")
            print("Train var: {} \nTrain Bias2: {}\nTrain Loss: {}".format(train_var_sum, train_bias2_sum, train_loss_sum))

    stats = {
        "test_loss": test_loss_sum,
        "test_acc": test_acc_sum,
        "test_var": test_var_sum,
        "test_bias2": test_bias2_sum,
        "train_loss": train_loss_sum,
        "train_acc": train_acc_sum,
        "train_var": train_var_sum,
        "train_bias2": train_bias2_sum,
    }
    print("Stats Dict: ", stats)

    avg_test_loss, avg_test_acc, avg_test_var, avg_test_bias2, avg_train_loss, avg_train_acc, avg_train_var, avg_train_bias2 = get_stats(stats)

    utils.write_pkl(out_path, stats)

    return True, [avg_train_acc, avg_test_acc, avg_train_loss, avg_test_loss, \
                  avg_train_var, avg_test_var, \
                  avg_train_bias2, avg_test_bias2, \
                  avg_train_acc - avg_test_acc, avg_test_loss - avg_train_loss, \
                  detailed_losses]


if __name__  == "__main__":
    GPU_ID = sys.argv[4]
    assert(int(GPU_ID)>= 0)

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    data_dir = sys.argv[1]
    setup_dir_path = sys.argv[2]
    FORCE_STAT_REFRESH = True if sys.argv[3].lower() == "force" else False
    failed_stat_log_path = None
    if len(sys.argv) == 6:
        failed_stat_log_path = sys.argv[5]
        print("Logging bad stats in file: ", failed_stat_log_path)
    analyze_each_setup(data_dir, setup_dir_path, FORCE_STAT_REFRESH, failed_stat_log_path)

