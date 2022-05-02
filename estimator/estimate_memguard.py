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
import math

# FORCE_MEMGUARD_REFRESH = False # set this to True if you want to force to recompute all the stats for the memguard
TOTAL_BAD_SAMPLES = 0

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


################################################################################################################
################################### Helper Function s###########################################################
################################################################################################################


def calc_kl_div(p,q):
    return (p * (p / q).log()).sum()

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

def log_mse(log_path, repeat_num, model_num, loss, acc, variance, bias_square, unbiased_variance, unbiased_bias_square):
    log_str = "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f" % (
        repeat_num, model_num, loss, acc, variance, bias_square, unbiased_variance, unbiased_bias_square
    )
    #print(log_str)
    utils.write_txt(log_path, [log_str+"\n"])

def log_ce(log_path, repeat_num, model_num, loss, acc, variance, bias_square):
    log_str = "%d\t%d\t%f\t%f\t%f\t%f" % (
        repeat_num, model_num, loss, acc, variance, bias_square
    )
    #print(log_str)
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


def get_model_paths_dicts(model_dir, model_num_per_repeat,  N=3):
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

def get_stats(stats_dict):
    avg_test_loss = torch.mean(torch.tensor(stats_dict["test_loss"]))
    avg_test_acc = torch.mean(torch.tensor(stats_dict["test_acc"]))
    avg_test_var = torch.mean(torch.tensor(stats_dict["test_var"]))
    avg_test_bias2 = torch.mean(torch.tensor(stats_dict["test_bias2"]))

    avg_train_loss = torch.mean(torch.tensor(stats_dict["train_loss"]))
    avg_train_acc = torch.mean(torch.tensor(stats_dict["train_acc"]))

    return avg_test_loss, avg_test_acc, avg_test_var, avg_test_bias2, avg_train_loss, avg_train_acc


################################################################################################################
################################################################################################################


def compute_test_avg_log_output(data_size, data_loader, model_config, model_path_array_dict, memguard_outputs_dict_test):
    # for cross entropy loss
    # compute log-output average
    # logprobs_sum = torch.Tensor(test_size, int(model_config["out_dim"])).zero_().cuda()
    logprobs_sum = torch.Tensor(data_size, int(model_config["out_dim"])).zero_().type(torch.DoubleTensor).cuda()
    model_num = 0
    for repeat_id in range(len(model_path_array_dict)):
        model_paths_dict = model_path_array_dict[repeat_id]
        for model_id in model_paths_dict:
            if not model_id in memguard_outputs_dict_test:
                continue
            for outputs in memguard_outputs_dict_test[model_id]:
                outputs = function.softmax(outputs, dim=1)
                model_num += 1  # each output list is treated as if it corresponds to a new model, as memguard prediction vectors are different
                model_path = model_paths_dict[model_id]
                logprobs_sum += outputs.log()
    logprobs_avg = logprobs_sum / float(model_num)
    normalized_avg = compute_normalized_kl(logprobs_avg, int(model_config["out_dim"]), data_size)
    return normalized_avg

################################################################################################################
################################### COMPUTE BIAS AND VARIANCE FOR MSE (TEST) ###################################
################################################################################################################

def estimate_mse_test(model_config, data_size, data_loader, model_paths_dict, model_num_per_repeat, output_collection, norm_square_output_collection, cnt_element, loss_per_element, memguard_outputs_dict):
    class_num = int(model_config["out_dim"])
    acc_collection = 0
    detailed_losses = []
    model_num = 0
    for model_id in model_paths_dict:
        if not model_id in memguard_outputs_dict:
            continue
        model_path = model_paths_dict[model_id]
        for outputs in memguard_outputs_dict[model_id]:
            model_num += 1  # each output list is treated as if it corresponds to a new model, as memguard prediction vectors are different
            output_collection, norm_square_output_collection, loss_per_element, acc_collection, loss = compute_bias_variance_mse(class_num, data_loader, output_collection, norm_square_output_collection, loss_per_element, acc_collection, model_num, outputs)
            cnt_element += 1 # as each model is evaluated on all testing samples
            detailed_losses.append(loss)
    return output_collection, norm_square_output_collection, cnt_element, loss_per_element, acc_collection/model_num, detailed_losses


def compute_bias_variance_mse(class_num, data_loader, output_collection, norm_square_output_collection, loss_per_element, acc_collection, model_num, memguard_outputs):
    # Set to be evaluation model
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
            outputs = memguard_outputs[cnt: (cnt + input_num)]
            tmp = utils.criterion_list["mse"](outputs, labels_onehot)
            # Compute accuracy
            _, predictions = outputs.max(1)
            correct_num += predictions.eq(labels).sum().item()
            # Sum_{x,y}[(f(x) - y)^2]
            loss += tmp.item() * outputs.numel() # numel() give you the number of element

            loss_per_element[cnt: (cnt + input_num)] += tmp.item()*class_num # add the mean loss to every position corresponding to the given elements via idx_list
            # Sum[f(x)]
            output_collection[cnt: (cnt + input_num), :] += outputs
            # E_{D}Sum_{x,y}[(f_bar(x) - y)^2]
            bias_square += (output_collection[cnt: (cnt + input_num), :]/model_num - labels_onehot).norm() ** 2.0
            # Sum[f^2(x)]
            norm_square_output_collection[cnt: (cnt + input_num)] += outputs.norm(dim=1) ** 2.0
            # Var(f(x)) = E[(f(x) - f_bar(x))^2] = E[f^2(x)] - (E[f(x)])^2
            # E_{D}Sum_{x}[(f(x) - f_bar(x))^2] = Sum[f^2(x)] - Sum[(E[f(x)])^2]
            cnt += input_num
    # percentage
    acc_collection = acc_collection +  correct_num / float(cnt)
    return output_collection, norm_square_output_collection, loss_per_element, acc_collection, loss/cnt

################################################################################################################
################################### COMPUTE ACCURACY AND LOSS FOR MSE (TRAIN) ##################################
################################################################################################################

def estimate_mse_train_acc_loss(model_config, data_size, data_loader, model_paths_dict, repeat_id, model_num_per_repeat, data_idx_array_repeat, memguard_outputs_dict):
    class_num = int(model_config["out_dim"])
    loss_collection = 0
    acc_collection = 0
    detailed_losses = []
    model_num = 0 
    for model_id in model_paths_dict:
        if not model_id in memguard_outputs_dict:
            continue
        for memguard_outputs in memguard_outputs_dict[model_id]:
            model_num += 1
            model_path = model_paths_dict[model_id]
            model_index_for_current_repeat = model_id - model_num_per_repeat*repeat_id - 1 # the data_laoder and data_idx_array are for each repeat, so we need to do this subtraction
            data_loader_for_model = data_loader[model_index_for_current_repeat]
            data_idx_for_model = data_idx_array_repeat[model_index_for_current_repeat]
            loss, acc_collection = compute_acc_loss_mse_train(data_loader_for_model, data_idx_for_model, class_num, acc_collection, memguard_outputs)
            loss_collection += loss
            detailed_losses.append(loss)
    return loss_collection/model_num, acc_collection/model_num, detailed_losses

def compute_acc_loss_mse_train(data_loader, idx_list, class_num, acc_collection, memguard_outputs):
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
            outputs = memguard_outputs[cnt: cnt+input_num]
            # Compute loss which is averaged over each element (note: not just over each x)
            tmp = utils.criterion_list["mse"](outputs, labels_onehot)
            # Compute accuracy
            _, predictions = outputs.max(1)
            correct_num += predictions.eq(labels).sum().item()
            # Sum_{x,y}[(f(x) - y)^2]
            loss += tmp.item() * outputs.numel()
            cnt += input_num
    acc_collection = acc_collection + correct_num / float(cnt)
    return loss/cnt, acc_collection

################################################################################################################
################################################################################################################


################################################################################################################
################################### COMPUTE ACCURACY AND LOSS FOR CE (TRAIN) ###################################
################################################################################################################

def estimate_ce_train_acc_loss(data_size, data_loaders, model_paths_dict, repeat_id, model_num_per_repeat, data_idx_array_repeat, memguard_outputs_dict):
    model_num = 0
    loss_sum = 0.0
    acc_sum = 0.0
    detailed_losses = []
    for model_id in model_paths_dict:
        if model_id not in memguard_outputs_dict:
            continue
        for memguard_outputs in memguard_outputs_dict[model_id]:
            model_num += 1
            model_path = model_paths_dict[model_id]
            model_index_for_current_repeat = model_id - model_num_per_repeat*repeat_id - 1 # the data_laoder and data_idx_array are for each repeat, so we need to do this subtraction
            loss, acc = compute_acc_loss_ce_train(data_loaders[model_index_for_current_repeat], data_idx_array_repeat[model_index_for_current_repeat], memguard_outputs)
            loss_sum += loss
            acc_sum += acc
            detailed_losses.append(loss)
    return loss_sum/model_num, acc_sum/model_num, detailed_losses


def compute_acc_loss_ce_train(data_loader, data_idx, memguard_outputs):
    test_loss = 0
    correct_num = 0
    cnt = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            input_num = labels.size(0)
            outputs = memguard_outputs[cnt: (cnt + input_num)]
            batch_loss = criterion_list["ce"](outputs, labels)
            test_loss += batch_loss.item() * input_num
            _, preds = outputs.max(1)
            correct_num += preds.eq(labels).sum().item()
            cnt += input_num
    return test_loss/cnt, correct_num/float(cnt)

################################################################################################################
################################################################################################################


################################################################################################################
################################### COMPUTE BIAS-VARIANCE FOR CE (TEST) ########################################
################################################################################################################


def compute_bias_variance_ce(data_loader, normalized_avg, memguard_outputs):
    #model.eval()
    global TOTAL_BAD_SAMPLES
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
            outputs = memguard_outputs[cnt: (cnt + input_num)]
            batch_loss = utils.criterion_list["ce"](outputs, labels)
            test_loss += batch_loss.item() * input_num
            _, preds = outputs.max(1)
            correct_num += preds.eq(labels).sum().item()
            outputs = function.softmax(outputs, dim=1)
            bias2 += nll_loss(normalized_avg[cnt:cnt+input_num, :].log(), labels)
            for sample_idx in range(input_num):
                sample_var = calc_kl_div(normalized_avg[cnt+sample_idx], outputs[sample_idx])
                #pdb.set_trace()
                if (math.isnan(sample_var) or sample_var <= - 0.0001):
                    TOTAL_BAD_SAMPLES += 1
                    continue
                variance += sample_var
            cnt += input_num
    return test_loss/cnt, correct_num/float(cnt), bias2/cnt, variance/cnt

def estimate_ce_test(model_config, data_loader, model_paths_dict, normalized_avg, repeat_id, model_num_per_repeat, memguard_outputs_dict):
    model_num = 0
    variance_sum = 0.0
    loss_sum = 0.0
    acc_sum = 0.0
    bias2_sum = 0.0
    detailed_losses = []
    for model_id in model_paths_dict:
        if not model_id in memguard_outputs_dict:
            # MEMGUARD: we do not do anything for the models for which we don't have memguard data
            continue
        model_path = model_paths_dict[model_id]
        print("Debug: Working on outputs of model: {}".format(model_path))
        for outputs in memguard_outputs_dict[model_id]:
            model_num += 1  # each output list is treated as if it corresponds to a new model, as memguard prediction vectors are different
            loss, acc, bias2, variance = compute_bias_variance_ce(data_loader, normalized_avg, outputs)
            variance_sum += variance
            loss_sum += loss
            acc_sum += acc
            bias2_sum += bias2
            detailed_losses.append(loss)
    return loss_sum/model_num, acc_sum/model_num, variance_sum/model_num, bias2_sum/model_num, detailed_losses

################################################################################################################
################################################################################################################


################################################################################################################
################################### MAIN FUNCTION TO BE CALLED FOR EACH SETUP ##################################
################################################################################################################

def analyze_each_setup_memguard(data_dir, setup_dir_path, FORCE_MEMGUARD_REFRESH = False, NO_EVALUATION = False):
    """
    Analyze each setup of memguard.
    FORCE_MEMGUARD_REFRESH: If set True, it will force refreshing the stats even if they exists.
    NO_EVALUATION: If set True, it will disable evaluation of stats irrespective of whether it exists. It will overwrite FORCE_MEMGUARD_REFRESH. 
    """
    # Collect all the updated memguard output data
    config_path = os.path.join(setup_dir_path, "config.ini")
    if not os.path.exists(config_path):
        print("Error: Configuration file does not exist ... %s" % config_path)
        return False, None


    out_path = os.path.join(setup_dir_path, "memguard_stats.pkl")
    if not FORCE_MEMGUARD_REFRESH and os.path.exists(out_path):
        stats = utils.read_pkl(out_path)
        avg_test_loss, avg_test_acc, avg_test_var, avg_test_bias2, avg_train_loss, avg_train_acc = get_stats(stats)
        print("Loaded computed memguard stats ...")
        return True, [float(avg_train_acc), float(avg_test_acc), float(avg_train_loss), float(avg_test_loss), \
                  float(avg_test_var), \
                  float(avg_test_bias2), \
                  float(avg_train_acc - avg_test_acc), float(avg_test_loss - avg_train_loss), \
                  None]

    if NO_EVALUATION:
        return False, None
   
    train_config, model_config = train.load_config(config_path)
    memguard_outputs_dict_test = dict()
    memguard_outputs_dict_train = dict()

    memguard_base = os.path.join(setup_dir_path, "memguard")
    if not os.path.exists(memguard_base):
        print("Error: No memguard data")
        return None, []
    stat_dict = {}
    defence_files = os.listdir(memguard_base)
    for defence_file in defence_files:
        memguard_file = os.path.join(memguard_base, defence_file)
        data = np.load(memguard_file)
        if train_config["loss"] == "mse":
            # for MSE
            updated_member_after_softmax = data["updated_member_after_softmax"]
            updated_nonmember_after_softmax = data["updated_nonmember_after_softmax"]
        else:
        # for CE 
            updated_member_before_softmax = data["updated_member_before_softmax"]
            updated_nonmember_before_softmax = data["updated_nonmember_before_softmax"]

        model_id = get_target_model_id_from_defence_file_name(defence_file)

        if model_id == -1:
            print("Error: Cannot read target_model_id for the defence file: ", defence_file)
            continue

        if model_id not in memguard_outputs_dict_test:
            memguard_outputs_dict_test[model_id] = []
            memguard_outputs_dict_train[model_id] = []

        if train_config["loss"] == "mse":
            memguard_outputs_dict_test[model_id].append(updated_nonmember_after_softmax)
            memguard_outputs_dict_train[model_id].append(updated_member_after_softmax)
        else:
            memguard_outputs_dict_test[model_id].append(updated_nonmember_before_softmax)
            memguard_outputs_dict_train[model_id].append(updated_member_before_softmax)

    # now we have all the memguard outputs for training and testing, we can get the stats
    return analyze_each_setup(data_dir, setup_dir_path, memguard_outputs_dict_test, memguard_outputs_dict_train, FORCE_MEMGUARD_REFRESH)

def get_target_model_id_from_defence_file_name(defence_file):
    # memguard_file = "/mnt/archive0/shiqi/epoch_500/with_scheduler-mse/cifar10-densenet161-w_2-ntrain_1k-N_3/memguard/cifar10-densenet161-w_2-1k-target_model_id_14-shadow_id_33-nm_model_id_9-ml_leaks-shadow-loss_mse-attack_model.npz"
    lst = defence_file.split("-")
    for k in lst:
        if k.find("target_model_id") != -1:
            return int(k.split("_")[-1])
    return -1
        

def analyze_each_setup(data_dir, setup_dir_path, memguard_outputs_dict_test, memguard_outputs_dict_train, FORCE_MEMGUARD_REFRESH = False):

    print("DEBUG: Target Model Id's for Memguard Stat Computation:")
    print("\tTarget Id\tNumber of output lists")
    for i in memguard_outputs_dict_test:
        print("\t{}\t{}".format(i, len(memguard_outputs_dict_test[i])))
        memguard_outputs_dict_test[i] = list(map(lambda o: torch.from_numpy(o).float().cuda(), memguard_outputs_dict_test[i]))
        memguard_outputs_dict_train[i] = list(map(lambda o: torch.from_numpy(o).float().cuda(), memguard_outputs_dict_train[i]))
    # load config
    config_path = os.path.join(setup_dir_path, "config.ini")
    if not os.path.exists(config_path):
        print("Error: Configuration file does not exist ... %s" % config_path)
        return False, None
    train_config, model_config = train.load_config(config_path)
    out_path = os.path.join(setup_dir_path, "memguard_stats.pkl")
    if not FORCE_MEMGUARD_REFRESH and os.path.exists(out_path):
        stats = utils.read_pkl(out_path)
        avg_test_loss, avg_test_acc, avg_test_var, avg_test_bias2, avg_train_loss, avg_train_acc = get_stats(stats)
        print("Loaded computed memguard stats ...")
        return True, [avg_train_acc, avg_test_acc, avg_train_loss, avg_test_loss, \
                  avg_test_var, \
                  avg_test_bias2, \
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
    model_path_array_dict = get_model_paths_dicts(model_dir, model_num_per_repeat, split_num)
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
    detailed_losses = {}

    if train_config["loss"] == "ce":
        test_normalized_avg = compute_test_avg_log_output(test_size, test_loader, model_config, model_path_array_dict, memguard_outputs_dict_test) #, model_num_per_repeat)
        print("Debug: Total bad samples: {}".format(TOTAL_BAD_SAMPLES))
    else:
        test_normalized_avg = None
    detailed_losses = {}
    class_num = int(model_config["out_dim"])
    if train_config["loss"] == "mse":
        # go through all the model_ids, and keep udpating the following data structures, finally we will compute the stats
        output_collection = torch.Tensor(test_size, class_num).zero_().cuda()
        norm_square_output_collection = torch.Tensor(test_size).zero_().cuda()
        cnt_element = torch.Tensor(test_size).zero_().type(torch.DoubleTensor).cuda() # this will store the number of times we update/add to the output_collection and norm_square_output_collection 
        loss_per_element = torch.Tensor(test_size).zero_().type(torch.DoubleTensor).cuda() # this will store the total loss of each element by adding to the loss to corresponding location every time we process that element through a model

        for repeat_id in range(split_num):
            print("Debug: Start repeat %d" % repeat_id)
            model_paths_dict = model_path_array_dict[repeat_id]
            intersect = set(model_paths_dict.keys()).intersection(set(memguard_outputs_dict_test.keys()))
            if not intersect:
                print("Debug: No memguard data for current repeat. Skipping...")
                continue
            output_collection, norm_square_output_collection, cnt_element, loss_per_element, test_acc, test_detail_losses = estimate_mse_test(model_config, test_size, test_loader, model_paths_dict, model_num_per_repeat, output_collection, norm_square_output_collection, cnt_element, loss_per_element,  memguard_outputs_dict_test)
            detailed_losses[repeat_id] = {"test": test_detail_losses}
            test_acc_sum.append(float(test_acc))

            # now we process training for accuracy and loss
            train_loaders = train_loader_array[repeat_id]
            data_idx_array_repeat = train_idx_array[repeat_id]
            train_loss, train_acc, train_detail_losses = estimate_mse_train_acc_loss(model_config, train_size, train_loaders, model_paths_dict, repeat_id, model_num_per_repeat, data_idx_array_repeat, memguard_outputs_dict_train)
            detailed_losses[repeat_id]["train"] = train_detail_losses
            train_acc_sum.append(train_acc)
            train_loss_sum.append(train_loss)

        # compute stats using the output_collection, norm_square_output_collection, cnt_element, and loss_per_element
        avg_loss_per_element = loss_per_element/cnt_element
        avg_norm_square_output_collection  = norm_square_output_collection/cnt_element
        avg_output_collection  = torch.div(output_collection, cnt_element[:, None])

        # to compute the number of elements that are involved in the computation of variance/bias, we find the number of elements that had [cnt_element >2]
        test_var = (avg_norm_square_output_collection.sum() - (avg_output_collection.norm() ** 2))/test_size
        total_loss = sum(avg_loss_per_element)/test_size

        test_bias2 = total_loss - test_var
        # now we append these single values in the list for consistency with other code
        test_loss_sum.append(float(total_loss))
        test_var_sum.append(float(test_var))
        test_bias2_sum.append(float(test_bias2))
        print("Debug: Finish computing stats on training samples")
        print("Debug: Test var: {} \nTest Bias2: {}\nTest Loss: {}".format(test_var_sum, test_bias2_sum, test_loss_sum))

    if train_config["loss"] == "ce":
        for repeat_id in range(split_num):
            # memguard_output dicts are without softmax
            print("Start repeat %d" % repeat_id)
            print("Start: Testing samples")
            model_paths_dict = model_path_array_dict[repeat_id]
            intersect = set(model_paths_dict.keys()).intersection(set(memguard_outputs_dict_test.keys()))
            if not intersect:
                print("Debug: No memguard data for current repeat. Skipping...")
                continue
            # start with test set
            test_loss, test_acc, test_var, test_bias2, test_detail_losses = estimate_ce_test(model_config, test_loader, model_paths_dict, test_normalized_avg, repeat_id, model_num_per_repeat, memguard_outputs_dict_test)
            detailed_losses[repeat_id] = {"test": test_detail_losses}
            test_loss_sum.append(float(test_loss))
            test_acc_sum.append(float(test_acc))
            test_var_sum.append(float(test_var))
            test_bias2_sum.append(float(test_bias2))
            print("Done: Testing samples")

            print("Start: Training samples")
            train_loaders = train_loader_array[repeat_id]
            data_idx_array_repeat = train_idx_array[repeat_id]
            train_loss, train_acc, train_detail_losses = estimate_ce_train_acc_loss(train_size, train_loaders, model_paths_dict, repeat_id, model_num_per_repeat, data_idx_array_repeat, memguard_outputs_dict_train)
            detailed_losses[repeat_id]["train"] = train_detail_losses
            train_acc_sum.append(train_acc)
            train_loss_sum.append(train_loss)
            print("Done: Training samples")
        print("Debug: Test var: {} \nTest Bias2: {}\nTest Loss: {}".format(test_var_sum, test_bias2_sum, test_loss_sum))

    stats = {
        "test_loss": test_loss_sum,
        "test_acc": test_acc_sum,
        "test_var": test_var_sum,
        "test_bias2": test_bias2_sum,
        "train_loss": train_loss_sum,
        "train_acc": train_acc_sum,
    }

    avg_test_loss, avg_test_acc, avg_test_var, avg_test_bias2, avg_train_loss, avg_train_acc = get_stats(stats)
    utils.write_pkl(out_path, stats)
    print("Stats Dict", stats)

    return True, [avg_train_acc, avg_test_acc, avg_train_loss, avg_test_loss, \
                  avg_test_var, \
                  avg_test_bias2, \
                  avg_train_acc - avg_test_acc, avg_test_loss - avg_train_loss, \
                  detailed_losses]

################################################################################################################
################################################################################################################


if __name__  == "__main__":
    GPU_ID = sys.argv[4]
    assert(int(GPU_ID)>= 0)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    data_dir = sys.argv[1]
    setup_dir_path = sys.argv[2]
    FORCE_MEMGUARD_REFRESH = True if sys.argv[3].lower() == "force" else False
    analyze_each_setup_memguard(data_dir, setup_dir_path, FORCE_MEMGUARD_REFRESH)

