"""
Shadow Model Attack
Reference: https://github.com/csong27/membership-inference/blob/a056ddc9599c1f7abd253722622f6b533b7c1925/attack.py#L85
"""
import sys
sys.path.append("../")

import os
import torch
import argparse
from trainer import models
import attack_models
from trainer import train
import torch.optim.lr_scheduler as lr_scheduler
import estimator.utils as utils
from estimator import estimate
import numpy as np
from tqdm import tqdm
import estimator.prepare_dataset as prep_data
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Shadow Models Attack")
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
        "--config_file", "-c", dest="config_file", type=str, required=True, help="Path to the configuration file of the shadow/target model & the training procedure "
    )
    parser.add_argument(
        "--model_dir", "-m", dest="model_dir", type=str, required=False, default="", help="Path to the directory containing models used as target and shadow model"
    )
    parser.add_argument(
        "--attack_config_file", "-ac", dest="attack_config_file", type=str, required=True, help="Path to the configuration file of the attack model & the training procedure"
    )
    parser.add_argument(
        "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
    )
    parser.add_argument(
        "--seeds", "-s", dest="seeds", type=str, default="0", required=False, help="The random seed for performing shadow model attack."
    )
    parser.add_argument(
        "--target_model_id", "-t", dest="target_model_id", type=int, required=False, default=1, help="The model id of the target model."
    )
    parser.add_argument(
        "--shadow_model_ids", "-l", dest="shadow_model_ids", type=str, required=False, default="", help="Id's of the model that are to be used as shadow models (e.g., `2,3`)."
    )
    parser.add_argument(
        "--attack_data_file", "-ad", dest="attack_data_file", type=str, required=False, default="", help="Path to attack data file"
    )
    parser.add_argument(
        "--save-freq", "-f", dest="save_freq", type=int, required=False, default=200, help="Save Frequency"
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    """
    Set random seed for numpy, torch and random
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class AttackDataset(Dataset):
    def __init__(self, X, Y ):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return (x,y)

def get_model_paths_dict(model_dir):
    """
    Returns dictionary dict with dict[model_id] = model_path for all models in model_dir
    """
    file_list = os.listdir(model_dir)
    file_dict = {}
    for file_name in file_list:
        if file_name[:11] == "final_model":
            file_id = int(file_name.split('.')[0].split("_")[-1])
            file_dict[file_id] = os.path.join(model_dir, file_name)
    return file_dict


def select_data(dataset, idx_list):
    """
    Returns a copy of data from dataset projected to indices from idx_list
    """
    dataset_copy = deepcopy(dataset)
    dataset_copy.data = [dataset_copy.data[idx] for idx in idx_list]
    dataset_copy.targets = [dataset_copy.targets[idx] for idx in idx_list]
    return dataset_copy

def get_attack_data(dataloader, net, num_classes, in_training):
    """
    The function applies the model net to the inputs of dataloader, and creates data for the attack model
    """
    X = torch.zeros([len(dataloader.dataset), num_classes], dtype=torch.float32,  requires_grad=False)
    if in_training:
        Y = torch.ones([len(dataloader.dataset)], dtype=torch.long,  requires_grad=False)
    else:
        Y = torch.zeros([len(dataloader.dataset)], dtype=torch.long,  requires_grad=False)
    classes = torch.zeros([len(dataloader.dataset)], dtype=torch.int32)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cuda()
            outputs = net(inputs)
            s = batch_idx*dataloader.batch_size
            t = s + len(outputs)
            X[s:t] = outputs
            classes[s:t] = labels
    return X,Y,classes

def get_attack_data_combined(test_loader, train_loader, model, num_classes):
    """
    Returns the prediction vector for train_loader, test_loader using model
    """
    attack_x_1, attack_y_1, classes_1 = get_attack_data(test_loader, model, num_classes, False)
    attack_x_2, attack_y_2, classes_2 = get_attack_data(train_loader, model, num_classes, True)
    attack_x = torch.cat([attack_x_1, attack_x_2], dim = 0)
    attack_y = torch.cat([attack_y_1, attack_y_2], dim = 0)
    classes = torch.cat([classes_1,classes_2], dim = 0)
    return (attack_x, attack_y, classes)

def get_optimizer_scheduler(net, train_config, model_config):
    """
    Function returns optimizer and scheduler for using train_config and model_config
    """
    if "momentum" in train_config: # no momentum for adam
        optimizer = train.optimizer_list[train_config["optimizer"]](
            net.parameters(),
            lr = float(train_config["lr"]),
            momentum = float(train_config["momentum"]),
            weight_decay = float(train_config["weight_decay"])
        )
    else:
        optimizer = train.optimizer_list[train_config["optimizer"]](
            net.parameters(),
            lr = float(train_config["lr"]),
            weight_decay = float(train_config["weight_decay"])
        )
    if "lr_decay" in train_config and "lr_step_size" in train_config:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(train_config["lr_step_size"]), gamma=float(train_config["lr_decay"]))
    else:
        scheduler = None
    return optimizer, scheduler



def get_shadow_model_ids_list(shadow_model_ids):
    if "~" in shadow_model_ids:
        tmp = shadow_model_ids.split("~")
        shadow_model_id_list = list(range(int(tmp[0]), int(tmp[1])+1))
    else:
        shadow_model_id_list = [int(tmp) for tmp in shadow_model_ids.split(",")]
    return shadow_model_id_list

def perform_attack(args, seed, attack_log_path):
    # load shadow/target model config
    if not os.path.exists(args.config_file):
        raise Exception("Error: Configuration file does not exist ... %s" % args.config_file)
    train_config, model_config = train.load_config(args.config_file)
    model_config["w_softmax"] = "yes"      # always use the softmax function on predictions

    # load attack model config
    if not os.path.exists(args.config_file):
        raise Exception("Error: Configuration file does not exist ... %s" % args.attack_config_file)
    attack_train_config, attack_model_config = train.load_config(args.attack_config_file)
    attack_model_dir = os.path.dirname(args.attack_config_file)
    # load data
    if not os.path.exists(args.idx_file):
        raise Exception("Error: Data idx file does not exist ... %s" % args.idx_file)

    train_set, test_set = train.load_complete_data(args.data_tag, args.data_dir)
    shadow_model_id_list = get_shadow_model_ids_list(args.shadow_model_ids)
    model_dir = os.path.dirname(args.config_file) if (args.model_dir == "") else args.model_dir
    idx_data = utils.read_npy(args.idx_file) # load idx data

    # prepare test & train loader
    test_loader = train.prepare_test_set(args.data_tag, test_set, idx_data["test_idx_set"], int(train_config["test_batch_size"]))

    # get the target and shodow model file paths
    model_dict = get_model_paths_dict(model_dir)
    target_model_file = model_dict[args.target_model_id]
    print("Target Model is: ", target_model_file)

    print("Following shadow models will be used:")
    [print("\t", model_dict[m_id]) for m_id in shadow_model_id_list]


    # prepare the testing data for the attack model
    if args.attack_data_file == "":
        num_classes = int(model_config["out_dim"])
        idx_target_train_indices = idx_data["train_idx_sets"][args.target_model_id-1]
        test_indices = idx_data["test_idx_set"]
        set_size = min(idx_target_train_indices.size, test_indices.size//2)  # to make sure we can select equal number of training and testing samples to create attack model training data
        idx_target_train_indices = idx_target_train_indices[:set_size]
        test_indices = test_indices[:2*set_size]  # to split the test into two parts each of size set_size
        idx_target_test_set, idx_shadow_test_set = torch.utils.data.random_split(test_indices, [set_size, set_size])
        print("Generating testing data for attack model for target model id: %d"% (args.target_model_id))
        train_set_copy = deepcopy(train_set)
        test_set_copy = deepcopy(test_set)
        target_test_loader = train.prepare_test_set(args.data_tag, test_set_copy, idx_target_test_set, int(train_config["test_batch_size"]))
        target_train_loader = train.prepare_test_set(args.data_tag, train_set_copy, idx_target_train_indices, int(train_config["test_batch_size"]))
        target_model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        target_model.load_state_dict(torch.load(target_model_file))
        test_attack_x, test_attack_y, test_classes = get_attack_data_combined(target_test_loader, target_train_loader, target_model, num_classes)
        print("\tTesting data generated!\n")

        # Generating training data for the attack model using the shadow models
        train_attack_x, train_attack_y, train_classes = [],[],[]
        test_set_copy = deepcopy(test_set)
        # the following test loader is common for all the shadow models
        shadow_test_loader = train.prepare_test_set(args.data_tag, test_set_copy, idx_shadow_test_set, int(train_config["test_batch_size"]))
        print("Generating training data for attack model using:")
        for m_id in shadow_model_id_list:
            print("\tShadow model id: %d"%(m_id))
            shadow_model = models.MODELS[model_config["model_tag"]](model_config).cuda()
            shadow_model.load_state_dict(torch.load(model_dict[m_id]))
            idx_shadow_train_indices = idx_data["train_idx_sets"][m_id-1][:set_size] # get the training data indices for the shadow model m_id
            train_set_copy = deepcopy(train_set)
            shadow_train_loader = train.prepare_test_set(args.data_tag, train_set_copy, idx_shadow_train_indices, int(train_config["train_batch_size"]))
            train_attack_x_i, train_attack_y_i, train_classes_i = get_attack_data_combined(shadow_test_loader, shadow_train_loader, shadow_model, num_classes)
            train_attack_x.append(train_attack_x_i)
            train_attack_y.append(train_attack_y_i)
            train_classes.append(train_classes_i)
        print("\tTraining data generated!")
        train_attack_x = torch.cat(train_attack_x, dim=0)
        train_attack_y= torch.cat(train_attack_y, dim=0)
        train_classes = torch.cat(train_classes, dim=0)

        attack_dict = {
            "test_attack_x" : test_attack_x,
            "test_attack_y" : test_attack_y,
            "train_attack_x" : train_attack_x,
            "train_attack_y" : train_attack_y,
            "test_classes" : test_classes,
            "train_classes" : train_classes,
        }
        file_path = os.path.join(attack_model_dir, "attack-data-tensor-dict_%s-target_model_id_%d-shadow_model_id_%s-seed_%s.pt" % (
            args.data_tag, args.target_model_id, args.shadow_model_ids,seed))
        torch.save(attack_dict, file_path)
        print("\tTraining data saved in file %s" % (file_path))
    else:
        if not os.path.exists(args.config_file):
            raise Exception("Error: Attack data file does not exist ... %s" % args.attack_config_file)

        attack_dict = torch.load(args.attack_data_file)
        test_attack_x = attack_dict["test_attack_x"]
        test_attack_y = attack_dict["test_attack_y"]
        train_attack_x = attack_dict["train_attack_x"]
        train_attack_y = attack_dict["train_attack_y"]
        test_classes = attack_dict["test_classes"]
        train_classes = attack_dict["train_classes"]
        print("Attack model data loaded from the file.")



    # we create attack model for each class and train them
    train_batch_size = int(attack_train_config["train_batch_size"])
    test_batch_size = int(attack_train_config["test_batch_size"])

    attack_model = {}
    baseline_accuracy, attack_accuracy= [],[]
    num_classes = int(model_config["out_dim"])
    for class_num in range(num_classes):  
        print("*"*10,"\nTraining the attack model for class num: %d" % class_num)
        log_path = os.path.join(attack_model_dir, "log-class_num_%d-seed_%d.txt" % (class_num, seed))
        # initialize the attack model
        attack_model[class_num] = attack_models.MODELS[attack_model_config["model_tag"]](attack_model_config).cuda() 
        test_attack_x_i, test_attack_y_i = test_attack_x[test_classes==class_num], test_attack_y[test_classes==class_num]
        train_attack_x_i, train_attack_y_i = train_attack_x[train_classes==class_num], train_attack_y[train_classes==class_num]

        # printing stat about the current class instances
        tr = len(train_attack_x_i)
        ts = len(test_attack_x_i)
        trm = (train_attack_y_i==1).sum().item()   # attack training which were in_training for shadows
        tsm = (test_attack_y_i==1).sum().item()    # attack training which were in_training for the target
        trc = torch.argmax(train_attack_x_i, dim=1).eq(class_num).sum().item()   # attack training which were correctly classified by the shadow models
        tsc = torch.argmax(test_attack_x_i, dim=1).eq(class_num).sum().item()    # attack training which were correctly classified by the target model 
        trmc = torch.argmax(train_attack_x_i[(train_attack_y_i==1).flatten()], dim=1).eq(class_num).sum().item()     # attack training which were member and were correctly classified by the shadows
        tsmc = torch.argmax(test_attack_x_i[(test_attack_y_i==1).flatten()], dim=1).eq(class_num).sum().item()       # attack testing which were member and were correctly classified by the target
        print("Attack training data stats:\n\tTotal size: %d\n\tIn_training (of shadow models): %d\n\tCorrectly classified: %d\n\tIn_training and correctly classified: %d"%(tr, trm, trc, trmc))
        print("Attack testing data stats:\n\tTotal size: %d\n\tIn_training (of target model): %d\n\tCorrectly classified: %d\n\tIn_training and correctly classified: %d"%(ts, tsm, tsc, tsmc))
        print("Baseline attack accuracy:\n\tTraining data of target model: %.4f\n\tTesting data of target model: %.4f"%((tsmc/tsm)*100, 100*(ts - tsm - (tsc - tsmc))/(ts - tsm)))
        baseline_accuracy.append(((tsmc/tsm)*100, 100*(ts - tsm - (tsc - tsmc))/(ts - tsm)))
        
        # initialize optimizer and scheduler
        optimizer, scheduler = get_optimizer_scheduler(attack_model[class_num], attack_train_config, attack_model_config)
        
        # Create datasets
        train_set = AttackDataset(train_attack_x_i, train_attack_y_i)

        target_testing_output = test_attack_x_i[(test_attack_y_i==0).flatten()]
        target_training_output = test_attack_x_i[(test_attack_y_i==1).flatten()]
        test_set_train = AttackDataset(target_training_output, torch.ones([len(target_training_output)], dtype=torch.long,  requires_grad=False))
        test_set_test = AttackDataset(target_testing_output, torch.zeros([len(target_testing_output)], dtype=torch.long,  requires_grad=False))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
        train_eval_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size)
        test_loader_train = torch.utils.data.DataLoader(test_set_train, batch_size=test_batch_size, shuffle=False)
        test_loader_test = torch.utils.data.DataLoader(test_set_test, batch_size=test_batch_size, shuffle=False)
        header = "#Epoch\tTrainAcc\tTestAccTrain\tTestAccTest\tTrainLoss\tTestLossTrain\tTestLossTest"
        utils.write_txt(log_path, [header])
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean").cuda()
        print(header)
        for epoch_no in range(1, int(attack_train_config["epoch_num"])+1):
            # train the attack model
            train_fn(train_loader, attack_model[class_num], optimizer, loss_fn)
            train_loss, train_acc = test_fn(train_loader, attack_model[class_num], loss_fn) 
            test_loss_test, test_acc_test = test_fn(test_loader_test, attack_model[class_num], loss_fn)  # attack accuracy over target testing data
            test_loss_train, test_acc_train = test_fn(test_loader_train, attack_model[class_num], loss_fn)     # attack accuracy over target training data
            train_loss, train_acc = test_fn(train_eval_loader, attack_model[class_num], loss_fn)
            log_data = "%d\t%f\t%f\t%f\t%f\t%f\t%f" % (epoch_no, train_acc, test_acc_train, test_acc_test, train_loss, test_loss_train, test_loss_test)
            utils.write_txt(log_path, [log_data+"\n"])
            print(log_data)
            if epoch_no % args.save_freq == 0:
                torch.save(
                    attack_model[class_num].state_dict(),
                    os.path.join(attack_model_dir, "model-epoch_%d-class-%d-seed_%d.pkl" % (epoch_no, class_num, seed))
                )
            if scheduler is not None:
                scheduler.step()
        torch.save(
            attack_model[class_num].state_dict(), 
            os.path.join(attack_model_dir, "final_attack_model-class_%d-seed_%d.pkl" % (class_num, seed))
        )
        attack_accuracy.append((test_acc_train, test_acc_test))


    print("*"*10, "Summary", "*"*10)
    baseline_test = 0
    baseline_train = 0
    attack_test = 0
    attack_train = 0
    line = "Seed ClassNum BaselineTestAcc BaselineTrainAcc AttackTestAcc AttackTrainAcc Baseline(0.5) Attack(0.5)"
    print(line)
    for i in range(num_classes):
        line = "%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t"%(seed, i+1, baseline_accuracy[i][1],baseline_accuracy[i][0],attack_accuracy[i][1], attack_accuracy[i][0], (baseline_accuracy[i][1] + baseline_accuracy[i][0])/2, (attack_accuracy[i][1] + attack_accuracy[i][0])/2)
        print(line)
        utils.write_txt(attack_log_path, [line + "\n"])
        baseline_train += baseline_accuracy[i][0]
        baseline_test += baseline_accuracy[i][1]
        attack_train += attack_accuracy[i][0]
        attack_test += attack_accuracy[i][1]

    print("Overall:")
    print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t"%(baseline_test/num_classes, baseline_train/num_classes, attack_test/num_classes, attack_train/num_classes, (baseline_test/num_classes+baseline_train/num_classes)/2, (attack_test/num_classes+ attack_train/num_classes)/2))

def train_fn(train_loader, net, optimizer, loss_fn):
    net.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


def test_fn(test_loader, net, loss_fn):
    net.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predictions = outputs.max(1)
            correct += predictions.eq(labels).sum().item()
            cnt += labels.size(0)
    return test_loss/cnt, 100.0 * correct/cnt


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    line = "Seed ClassNum BaselineTestAcc BaselineTrainAcc AttackTestAcc AttackTrainAcc Baseline(0.5) Attack(0.5)"
    attack_model_dir = os.path.dirname(args.attack_config_file)
    attack_log_path = os.path.join(attack_model_dir, "shadow_attack_summary.csv")
    utils.write_txt(attack_log_path, [line + "\n"])
    for seed in args.seeds.split(","):
        seed = int(seed)
        set_seed(seed)
        perform_attack(args, seed, attack_log_path)


if __name__ == '__main__':
    main()