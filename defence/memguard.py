
'''
The evaluation of MemGuard requires 4 models:
- Target Model: Model for exploration 
- Defense Model: Attack model trained by defenders. Using the same dataset as the one used for training Target Model
- Shadow Model: Model trained by attackers having the same architecture as Target Model
- Attack Model: Attack model trained by attackers having a different architecture as Defense Model. Using the same dataset as the one used for training Shadow Model

'''

SchedulerMaps = {
    "w_scheduler": "with_scheduler",
    "wo_scheduler": "without_scheduler"
}
TrialNum = 10
DataSeed = 0
DataSplitNum=3
DataTestSize=10000
TrainSizeMap = {
    "1k": 1000,
    "5k": 5000
}
TOPN=3
MaxIteration=300
C1=1.0
C2=10.0
C3Initial=0.1
UpdateRate = 0.1

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import configparser
import argparse
import numpy as np
from estimator import estimate
from loader import utils
from trainer import train, models
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MemGuard")
    parser.add_argument(
        "--attack_model_dir", "-a", dest="attack_model_dir", type=str, required=True, help="The folders to the attack models trained by attackers"
    )
    parser.add_argument(
        "--defend_model_dir", "-d", dest="defend_model_dir", type=str, required=True, help="The folder to the defend models which are attack models which the defenders train using the target model as the shadow model"
    )
    parser.add_argument(
        "--target_model_dir", "-t", dest="target_model_dir", type=str, required=True, help="The folder to the target model of MI attacks"
    )
    parser.add_argument(
        "--data_dir", "-dd", dest="data_dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--dataset", "-da", dest="dataset", type=str, required=True, help="Which dataset was the target models trained on?"
    )
    parser.add_argument(
        "--epoch_num", "-e", dest="epoch_num", type=str, required=True, help="How long did you train the target models?"
    )
    parser.add_argument(
        "--scheduler", "-s", dest="scheduler", type=str, required=True, help="Did you use scheduler when training target models?"
    )
    parser.add_argument(
        "--loss", "-l", dest="loss", type=str, required=True, help="Which loss function did you use to train the target models"
    )
    parser.add_argument(
        "--arch", "-ar", dest="arch", type=str, required=True, help="Architectures of the target models"
    )
    parser.add_argument(
        "--width", "-w", dest="width", type=str, required=True, help="Widths of the target models"
    )
    parser.add_argument(
        "--train_size", "-ts", dest="train_size", type=str, required=True, help="The size of the training set used to train the target models"
    )
    parser.add_argument(
        "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
    )
    parser.add_argument(
        "--tuple_config_file", "-tc", dest="tuple_config_file", type=str, required=True, help="The file saving the tuples which are in the format of (target_model_path, attack_model_path, defence_model_path)"
    )
    args = parser.parse_args()
    return args

def check_dir_existence(dir_path):
    if not os.path.isdir(dir_path):
        raise Exception(f"ERROR: Cannot find the folder ... {dir_path}")

def group_files(attack_folder):
    filenames = os.listdir(attack_folder)
    file_dict = {}
    for filename in filenames:
        tmp = filename.split("-")
        width = int(tmp[2].split("_")[-1])
        train_size = tmp[3]
        target_model_id = int(tmp[4].split("_")[-1])
        shadow_model_id = int(tmp[5].split("_")[-1])
        nm_model_id = int(tmp[6].split("_")[-1])
        attack_type = tmp[7]
        if width not in file_dict:
            file_dict[width] = {}
        if train_size not in file_dict[width]:
            file_dict[width][train_size] = {}
        if attack_type not in file_dict[width][train_size]:
            file_dict[width][train_size][attack_type] = {}
        if target_model_id not in file_dict[width][train_size][attack_type]:
            file_dict[width][train_size][attack_type][target_model_id] = []
        file_dict[width][train_size][attack_type][target_model_id].append(
            [shadow_model_id, nm_model_id, os.path.join(attack_folder, filename)]
        )
    return file_dict

def process_args(args):
    check_dir_existence(args.attack_model_dir)
    check_dir_existence(args.defend_model_dir)
    check_dir_existence(args.target_model_dir)
    check_dir_existence(args.data_dir)

    epoch_nums = [int(item) for item in args.epoch_num.split(",")]
    datasets = args.dataset.split(",")
    schedulers = args.scheduler.split(",")
    losses = args.loss.split(",")
    archs = args.arch.split(",")
    widths = [int(item) for item in args.width.split(",")]
    train_sizes = args.train_size.split(",")

    models_info = {}
    if os.path.exists(args.tuple_config_file):
        models_info = utils.read_pkl(args.tuple_config_file)
    check_dir_existence(os.path.dirname(args.tuple_config_file))
    for epoch_num in epoch_nums:
        if epoch_num not in models_info:
            models_info[epoch_num] = {}
        for dataset in datasets:
            if dataset not in models_info[epoch_num]:
                models_info[epoch_num][dataset] = {}
            for scheduler in schedulers:
                if scheduler not in models_info[epoch_num][dataset]:
                    models_info[epoch_num][dataset][scheduler] = {}
                for loss in losses:
                    if loss not in models_info[epoch_num][dataset][scheduler]:
                        models_info[epoch_num][dataset][scheduler][loss] = {}
                    for arch in archs:
                        if arch not in models_info[epoch_num][dataset][scheduler][loss]:
                            models_info[epoch_num][dataset][scheduler][loss][arch] = {}
                        attack_model_subdir = f"{args.attack_model_dir}/epoch_{epoch_num}-with_top3/attack_models/{dataset}-{arch}-{scheduler}-{loss}"
                        defend_model_subdir = f"{args.defend_model_dir}/epoch_{epoch_num}-with_top3-defence/defence_models/{dataset}-{arch}-{scheduler}-{loss}"
                        check_dir_existence(attack_model_subdir)
                        check_dir_existence(defend_model_subdir)
                        attack_file_dict = group_files(attack_model_subdir)
                        defend_file_dict = group_files(defend_model_subdir)
                        for width in widths:
                            if width not in models_info[epoch_num][dataset][scheduler][loss][arch]:
                                models_info[epoch_num][dataset][scheduler][loss][arch][width] = {}
                            if width not in defend_file_dict:
                                print(f"WARN: Cannot find width={width} in {defend_model_subdir}")
                                continue
                            for train_size in train_sizes:
                                if train_size not in models_info[epoch_num][dataset][scheduler][loss][arch][width]:
                                    models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size] = {}
                                if train_size not in defend_file_dict[width]:
                                    print(f"WARN: Cannot find train_size={train_size} in {defend_model_subdir} with width={width}")
                                    continue
                                target_model_subdir = f"{args.target_model_dir}/epoch_{epoch_num}/{SchedulerMaps[scheduler]}-{loss}/{dataset}-{arch}-w_{width}-ntrain_{train_size}-N_3"
                                if not os.path.isdir(target_model_subdir):
                                    target_model_subdir = f"{target_model_subdir}-less_lr"
                                check_dir_existence(target_model_subdir)
                                for attack_type in attack_file_dict[width][train_size]:
                                    if attack_type not in models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size]:
                                        models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size][attack_type] = {}
                                    if attack_type not in defend_file_dict[width][train_size]:
                                        print(f"WARN: Cannot find attack_type={attack_type} in {defend_model_subdir} with width={width} and train_size={train_size}")
                                        continue
                                    for target_model_id in attack_file_dict[width][train_size][attack_type]:
                                        if target_model_id in models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size][attack_type]:
                                            print(f"INFO: Tuple information exists! epoch_num={epoch_num}, dataset={dataset}, loss={loss}, arch={arch}, width={width}, train_size={train_size}, attack_type={attack_type} and target_model_id={target_model_id}")
                                            continue
                                        if target_model_id not in defend_file_dict[width][train_size][attack_type]:
                                            print(f"WARN: Cannot find target_model_id={target_model_id} in {defend_model_subdir} with width={width}, train_size={train_size} and attack_type={attack_type}")
                                            continue
                                        attack_tmp = attack_file_dict[width][train_size][attack_type][target_model_id]
                                        defend_tmp = defend_file_dict[width][train_size][attack_type][target_model_id]
                                        model_num = len(attack_tmp)
                                        if model_num != len(defend_tmp):
                                            print(f"WARN: The number of defend models does not match with the number of attack models.\nDefence Models:\n{defend_tmp}\nAttack Models:\n{attack_tmp}")
                                            continue
                                        for item in defend_tmp:
                                            if item[0] != target_model_id:
                                                raise Exception(f"ERROR: The defend model should use the target model as the shadow model ... {item[-1]}")
                                            if item[1] == target_model_id:
                                                raise Exception(f"ERROR: The defend model should not label the training set of the target model as non-members and use it for training ... {item[-1]}")
                                        attack_idx_list = np.arange(len(attack_tmp))
                                        defend_idx_list = np.arange(len(attack_tmp))
                                        cnt = 0
                                        while(cnt < TrialNum):
                                            cnt += 1
                                            cont_tag = False
                                            for idx in range(model_num):
                                                if defend_tmp[defend_idx_list[idx]][1] == attack_tmp[attack_idx_list[idx]][0] or defend_tmp[defend_idx_list[idx]][1] == attack_tmp[attack_idx_list[idx]][1]:
                                                    # The defend model should not label any dataset used for training the attack model as non-members 
                                                    np.random.shuffle(defend_idx_list)
                                                    cont_tag = True
                                                    break
                                            if cont_tag == False:
                                                break
                                        if cont_tag:
                                            raise Exception(f"ERROR: The pair does not match!\nDefence Models:\n{defend_tmp}\nAttack Models:\n{attack_tmp}")
                                        target_model_path = f"{target_model_subdir}/models/final_model_{target_model_id}.pkl"

                                        models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size][attack_type][target_model_id] = []
                                        for idx in range(model_num):
                                            models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size][attack_type][target_model_id].append(
                                                {
                                                    "TargetModelPath": target_model_path,
                                                    "AttackModelPath": attack_tmp[attack_idx_list[idx]][-1],
                                                    "DefenceModelPath": defend_tmp[defend_idx_list[idx]][-1]
                                                }
                                            )
    # save tuple information
    utils.write_pkl(args.tuple_config_file, models_info)
    print(f"Saved the configuration file ... {args.tuple_config_file}")
    return models_info

def record_before_and_after_softmax_densenet_ce(model, data_loader):
    model.eval()
    before_softmax = []
    after_softmax = []
    label_collection = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            model_outputs = model(inputs)
            before_softmax.append(model_outputs)
            after_softmax.append(torch.softmax(model_outputs, axis = 1))
            label_collection.append(labels)
    return before_softmax, after_softmax, label_collection

def record_before_and_after_softmax_alexnetwol_mse(model, data_loader):
    model.eval()
    before_softmax = []
    after_softmax = []
    label_collection = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            model_outputs = model.net(inputs)
            before_softmax.append(model_outputs)
            after_softmax.append(torch.softmax(model_outputs, axis = 1))
            label_collection.append(labels)
    return before_softmax, after_softmax, label_collection

def record_before_and_after_softmax_resnet_mse(model, data_loader):
    model.eval()
    # register hook to record the output before softmax
    before_softmax = []
    def hook_before_softmax(module, fin, fout):
        before_softmax.append(fout)
    model.linear.register_forward_hook(hook_before_softmax)
    
    after_softmax = []
    label_collection = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            after_softmax.append(model(inputs))
            label_collection.append(labels)
    return before_softmax, after_softmax, label_collection

def record_before_and_after_softmax_densenet_mse(model, data_loader):
    model.eval()
    # register hook to record the output before softmax
    before_softmax = []
    def hook_before_softmax(module, fin, fout):
        before_softmax.append(fout)
    model.classifier.register_forward_hook(hook_before_softmax)
    
    after_softmax = []
    label_collection = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            after_softmax.append(model(inputs))
            label_collection.append(labels)
    return before_softmax, after_softmax, label_collection

def construct_defence_model_input_topn_wo_label(input, label):
    sorted_input, sorted_idx_list= torch.sort(input, descending=True)
    sorted_input = sorted_input[:TOPN]
    sorted_idx_list = sorted_idx_list[:TOPN]
    return sorted_input, sorted_idx_list

def construct_defence_model_input_topn_w_label(input, label):
    sorted_input, sorted_idx_list= torch.sort(input, descending=True)
    sorted_input = sorted_input[:TOPN]
    sorted_idx_list = sorted_idx_list[:TOPN]
    concated_input = torch.cat((sorted_input, torch.tensor([label]).cuda())) # topn_prediction_vector + [label]
    return concated_input, sorted_idx_list

def memguard(before_softmax, after_softmax, input_labels, defence_model, tag, construct_input_fn):
    success_cnt = 0
    output_vector_before_softmax = torch.zeros(before_softmax.shape, dtype=torch.float).cuda()
    output_vector_after_softmax = torch.zeros(after_softmax.shape, dtype=torch.float).cuda()
    sample_num = before_softmax.shape[0]
    for sample_idx in tqdm(range(sample_num), desc=f"[Memguard {tag}]"):
        # initialize with the original pred vector
        output_vector_before_softmax[sample_idx]=before_softmax[sample_idx]
        output_vector_after_softmax[sample_idx]=after_softmax[sample_idx]
        # get the original label
        origin_pred_label = torch.argmax(after_softmax[sample_idx])
        # get the onehot encoding of the original label
        origin_pred_label_onehot = torch.zeros(after_softmax[sample_idx].shape, dtype=torch.float).cuda()
        origin_pred_label_onehot[origin_pred_label] = 1.0
        # prepare the input for defence model
        defence_model_input, sorted_idx_list = construct_input_fn(after_softmax[sample_idx], input_labels[sample_idx])
        defence_model_input = defence_model_input.reshape(1, -1)
        # execute the defence model & get the score of being member
        member_prob = torch.softmax(defence_model(defence_model_input), axis = 1)[0][1] # member: class is 1
        # if the output score is already very close to 0.5, we can just use it for numerical reason
        if torch.abs(member_prob-0.5)<=1e-5: # memguard wants the member_prob to be close to 0.5 (random guess)
            success_cnt += 1
            continue
        # start searching for the adversarial example which can bypass the defence model with the same label outputted by the target model
        # weights for each loss
        c3 = C3Initial
        trial_cnt = 1
        while(c3 <= 100000.0):
            iteration_cnt = 1
            # prepare
            current_before_softmax = before_softmax[sample_idx].detach().clone()
            current_pred_label = origin_pred_label.detach().clone()
            current_member_prob = member_prob.detach().clone()
            # note that the attack model used by MLLeak is not the one with sigmoid at the end
            # hence, both the loss function and stopping criterion need to be adapted
            # - it does not exceed the max number of iterations
            # - the prediction label outputted by the target model does not change (which is the softmax(pred_vector))
            # - the prediction label from the defence model changes
            while(
                (iteration_cnt < MaxIteration) and \
                (
                    (origin_pred_label != current_pred_label) or \
                    ((member_prob-0.5)*(current_member_prob-0.5) > 0)
                )):
                current_before_softmax.requires_grad = True # memguard computes the gradient against the output of target model before softmax
                # prepare input for defence model
                current_after_softmax = torch.softmax(current_before_softmax, axis = 0)
                current_defence_model_input, current_sorted_idx_list = construct_input_fn(current_after_softmax, input_labels[sample_idx])
                
                # memguard wants the the probability of being a member to be 0.5
                loss1 = C1 * torch.abs(torch.softmax(defence_model(current_defence_model_input.reshape(1,-1)), axis = 1)[0][1] - 0.5) 
                # memguard wants to have the same prediction label outputted by the target model
                loss2 = C2 * (
                    torch.relu(
                        torch.max((1-origin_pred_label_onehot) * current_after_softmax - 1e8*origin_pred_label_onehot) - \
                        torch.sum(origin_pred_label_onehot * current_after_softmax)
                    )
                )
                # memguard wants to minimize the distortion on the prediction vector
                loss3 = c3 * torch.sum(
                    torch.abs(
                        current_after_softmax - after_softmax[sample_idx]    
                    )
                )
                loss = loss1 + loss2 + loss3
                # compute the gradient
                grad = torch.autograd.grad(loss, current_before_softmax)[0]
                # normalize the gradient
                normalized_grad = grad/torch.linalg.norm(grad)
                
                # update output of the target model before softmax
                current_before_softmax = (current_before_softmax - UpdateRate*normalized_grad).detach()
                # get the label of the updated result
                current_pred_label = torch.argmax(current_before_softmax)
                # get the output of defence model given the updated vector
                current_defence_model_input, current_sorted_idx_list = construct_input_fn(torch.softmax(current_before_softmax, axis = 0), input_labels[sample_idx])
                current_member_prob = torch.softmax(defence_model(current_defence_model_input.reshape(1,-1)), axis = 1)[0][1]
                iteration_cnt += 1
            if origin_pred_label != current_pred_label:
                if trial_cnt == 1:
                    print(f"WARN: Failed sample for label not same for id: {sample_idx},c3:{c3} not add noise")
                    success_cnt = success_cnt - 1
                break
            if (member_prob-0.5)*(current_member_prob-0.5) > 0:
                if trial_cnt == 1:
                    print(f"max iteration reached with id: {sample_idx}, max score: {torch.max(torch.softmax(current_before_softmax, axis =0))}, prediction_score: {current_member_prob}, c3: {c3}, not add noise")
                    success_cnt = success_cnt - 1
                break
            output_vector_before_softmax[sample_idx] = current_before_softmax.detach().clone()
            output_vector_after_softmax[sample_idx] = torch.softmax(output_vector_before_softmax[sample_idx], axis = 0)
            c3 = c3 * 10.0
            trial_cnt += 1
        success_cnt += 1
    print(f"WARN: Success Fraction: {success_cnt}/{sample_num}")
    return output_vector_before_softmax, output_vector_after_softmax

# def evaluate_memguard(origin_after_softmax, updated_after_softmax, attack_model):
#     defence_input_after_softmax = torch.zeros(origin_after_softmax.shape, dtype=torch.float).cuda()
#     np.random.seed(0)
#     for idx in range(origin_after_softmax.shape[0]):
#         if np.random.rand(1)<0.5:
#             defence_input_after_softmax[idx] = updated_after_softmax[idx]
#         else:
#             defence_input_after_softmax[idx] = origin_after_softmax[idx] # the original code has an issue at this point
#     attack_model()

def memguard_all_samples(dataset, target_model_id, target_model_path, attack_model_path, defense_model_path, output_path, data_dir, train_size, record_target_fn, construct_input_fn):
    if os.path.exists(output_path):
        print(f"{output_path} exists!")
        return 
    # load model configuration
    tagret_model_config_file = os.path.join(
        os.path.dirname(os.path.dirname(target_model_path)),
        "config.ini"
    )
    if not os.path.exists(tagret_model_config_file):
        raise Exception(f"ERROR: Cannot find the configuration file for the target model ... {target_model_path}")
    target_model_train_config, target_model_config = train.load_config(tagret_model_config_file)
    # load datasets
    # 1. load the evaluation dataset used for measuring the attack acc with memguard
    # training: the training set of target_model
    # testing: the whole testing dataset
    train_set, test_set = train.load_complete_data(dataset, data_dir)
    idx_file = os.path.join(data_dir, f"prepare_data-data_{dataset}-ntrial_{DataSplitNum}-train_size_{TrainSizeMap[train_size]}-test_size_10000-strategy_disjoint_split-seed_{DataSeed}.npz")
    if not os.path.exists(idx_file):
        raise Exception(f"ERROR: Cannot find the file ... {idx_file}")
    idx_data = utils.read_npy(idx_file)
    member_idx_data = idx_data["train_idx_sets"][target_model_id-1] # model_id start from 1 while the idx start from 0
    nonmember_idx_data = idx_data["test_idx_set"]
    member_loader = train.prepare_test_set(dataset, train_set, member_idx_data, int(target_model_train_config["test_batch_size"]))
    nonmeber_loader = train.prepare_test_set(dataset, test_set, nonmember_idx_data, int(target_model_train_config["test_batch_size"]))
    print("INFO: Loaded the evaluation dataset!")
    
    # load target model
    target_model = models.MODELS[target_model_config["model_tag"]](target_model_config).cuda()
    target_model.load_state_dict(torch.load(target_model_path))
    print("INFO: Loaded the target model!")
    # get the output after softmax and the output before softmax
    member_before_softmax, member_after_softmax, member_labels = record_target_fn(target_model, member_loader)
    member_before_softmax = torch.vstack(member_before_softmax)
    member_after_softmax = torch.vstack(member_after_softmax)
    member_labels = torch.cat(member_labels)

    nonmember_before_softmax, nonmember_after_softmax, nonmember_labels = record_target_fn(target_model, nonmeber_loader)
    nonmember_before_softmax = torch.vstack(nonmember_before_softmax)
    nonmember_after_softmax = torch.vstack(nonmember_after_softmax)
    nonmember_labels = torch.cat(nonmember_labels)
    
    # load defence model with two classes: member (class:1) & non-member (class:0)
    defence_model = torch.load(defense_model_path)
    print("INFO: Loaded the defence model!")
    
    # memguard
    updated_member_before_softmax, updated_member_after_softmax = memguard(member_before_softmax, member_after_softmax, member_labels, defence_model, "member", construct_input_fn)
    
    updated_nonmember_before_softmax, updated_nonmember_after_softmax = memguard(nonmember_before_softmax, nonmember_after_softmax, nonmember_labels, defence_model, "non-member", construct_input_fn)

    np.savez(
        output_path, 
        member_before_softmax = member_before_softmax.detach().cpu().numpy(),
        member_after_softmax = member_after_softmax.detach().cpu().numpy(),
        updated_member_before_softmax = updated_member_before_softmax.detach().cpu().numpy(),
        updated_member_after_softmax = updated_member_after_softmax.detach().cpu().numpy(),
        member_labels = member_labels.detach().cpu().numpy(),
        nonmember_before_softmax = nonmember_before_softmax.detach().cpu().numpy(),
        nonmember_after_softmax = nonmember_after_softmax.detach().cpu().numpy(),
        nonmember_labels = nonmember_labels.detach().cpu().numpy(),
        updated_nonmember_before_softmax = updated_nonmember_before_softmax.detach().cpu().numpy(),
        updated_nonmember_after_softmax = updated_nonmember_after_softmax.detach().cpu().numpy(),
    )
    

def test_memguard():
    dataset = "cifar10"
    target_model_id = 22
    target_model_path = "/mnt/archive0/shiqi/epoch_500/with_scheduler-mse/cifar10-densenet161-w_2-ntrain_5k-N_3/models/final_model_22.pkl"
    attack_model_path = "/home/hitarth/ml_leak_attacks/epoch_500-with_top3/attack_models/cifar10-densenet161-w_scheduler-mse/cifar10-densenet161-w_2-5k-target_model_id_22-shadow_id_29-nm_model_id_27-ml_leaks-shadow-loss_mse-attack_model.pth"
    defense_model_path = "/home/hitarth/ml_leak_attacks/epoch_500-with_top3-defence/defence_models/cifar10-densenet161-w_scheduler-mse/cifar10-densenet161-w_2-5k-target_model_id_22-shadow_id_22-nm_model_id_30-ml_leaks-shadow-loss_mse-attack_model.pth"
    data_dir = "/mnt/archive0/shiqi/datasets"
    train_size = "5k"
    record_target_fn = record_before_and_after_softmax_densenet_mse
    construct_input_fn = construct_defence_model_input_topn_wo_label
    memguard_all_samples(dataset, target_model_id, target_model_path, attack_model_path, defense_model_path, data_dir, train_size, record_target_fn, construct_input_fn)
    
RecordTargetFNList = {
    "densenet161": {
        "mse": record_before_and_after_softmax_densenet_mse,
        "ce": record_before_and_after_softmax_densenet_ce
    },
    "resnet34": {
        "mse": record_before_and_after_softmax_resnet_mse,
        "ce": record_before_and_after_softmax_densenet_ce # as same as densenet161
    },
    "alexnetwol": {
        "mse": record_before_and_after_softmax_alexnetwol_mse,
        "ce": record_before_and_after_softmax_densenet_ce # as same as densenet161
    }
}
ConstructInputFNList = {
    "ml_leaks": construct_defence_model_input_topn_wo_label,
    "ml_leaks_label": construct_defence_model_input_topn_w_label,
}
    

def memguard_all(models_info, args):
    data_dir = args.data_dir
    
    epoch_nums = [int(item) for item in args.epoch_num.split(",")]
    datasets = args.dataset.split(",")
    schedulers = args.scheduler.split(",")
    losses = args.loss.split(",")
    archs = args.arch.split(",")
    widths = [int(item) for item in args.width.split(",")]
    train_sizes = args.train_size.split(",")
    
    for epoch_num in epoch_nums:
        for dataset in models_info[epoch_num]:
            for scheduler in models_info[epoch_num][dataset]:
                for loss in models_info[epoch_num][dataset][scheduler]:
                    for arch in models_info[epoch_num][dataset][scheduler][loss]:
                        for width in models_info[epoch_num][dataset][scheduler][loss][arch]:
                            for train_size in models_info[epoch_num][dataset][scheduler][loss][arch][width]:
                                for attack_type in models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size]:
                                    for target_model_id in models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size][attack_type]:
                                        for item in models_info[epoch_num][dataset][scheduler][loss][arch][width][train_size][attack_type][target_model_id]:
                                            target_model_path = item["TargetModelPath"]
                                            attack_model_path = item["AttackModelPath"]
                                            defense_model_path = item["DefenceModelPath"]
                                            output_dir = os.path.join(
                                                os.path.dirname(os.path.dirname(target_model_path)), 
                                                "memguard"
                                            )
                                            if not os.path.isdir(output_dir):
                                                os.mkdir(output_dir)
                                            output_path = os.path.join(
                                                output_dir, 
                                                os.path.basename(attack_model_path)[:-4] + ".npz"
                                            )
                                            
                                            memguard_all_samples(dataset, target_model_id, 
                                                                 target_model_path, attack_model_path, defense_model_path, output_path,
                                                                 data_dir, train_size, 
                                                                 RecordTargetFNList[arch][loss], 
                                                                 ConstructInputFNList[attack_type])

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # test_memguard()
    # prepare all the configurations
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    models_info = process_args(args)
    memguard_all(models_info, args)

if __name__ == "__main__":
    main()
    
