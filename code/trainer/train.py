import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genericpath import samefile
import loader.prepare_dataset as prep_data
import torch
import trainer.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import configparser
import loader.utils as utils

optimizer_list = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}

def load_complete_data(data_tag, data_dir):
    dataloader = prep_data.RegisteredDatasets[data_tag]["loader"](data_dir)
    return dataloader.download()

def prepare_train_set(data_tag, train_set, idx_list, batch_size, shuffle_tag):
    train_set.data = [train_set.data[idx] for idx in idx_list]
    train_set.targets = [train_set.targets[idx] for idx in idx_list]
    train_set.transform = prep_data.RegisteredDatasets[data_tag]["train_transformer"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_tag)
    return train_loader

def prepare_test_set(data_tag, test_set, idx_list, batch_size):
    test_set.data = [test_set.data[idx] for idx in idx_list]
    test_set.targets = [test_set.targets[idx] for idx in idx_list]
    test_set.transform = prep_data.RegisteredDatasets[data_tag]["test_transformer"]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader

def train_mse(train_loader, net, class_num, optimizer):
    net.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        labels_onehot = torch.FloatTensor(labels.size(0), class_num).cuda()
        labels_onehot.zero_()
        labels_onehot.scatter_(1, labels.view(-1, 1).long(), 1)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = utils.criterion_list["mse"](outputs, labels_onehot)
        loss.backward()
        optimizer.step()

def test_mse(test_loader, net, class_num):
    net.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            labels_onehot = torch.FloatTensor(labels.size(0), class_num).cuda()
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.view(-1, 1).long(), 1)
            outputs = net(inputs)
            loss = utils.criterion_list["mse"](outputs, labels_onehot)
            test_loss += loss.item() * outputs.numel()
            _, predictions = outputs.max(1)
            correct += predictions.eq(labels).sum().item()
            cnt += labels.size(0)
    return test_loss/cnt, 100.0 * correct/cnt

def train_ce(train_loader, net, class_num, optimizer):
    net.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = utils.criterion_list["ce"](outputs, labels)
        loss.backward()
        optimizer.step()

def test_ce(test_loader, net, class_num):
    net.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = utils.criterion_list["ce"](outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predictions = outputs.max(1)
            correct += predictions.eq(labels).sum().item()
            cnt += labels.size(0)
    return test_loss/cnt, 100.0 * correct/cnt


def init_model_and_train(model_config, train_config):
    net = models.MODELS[model_config["model_tag"]](model_config).cuda()
    if "momentum" in train_config: # no momentum for adam
        optimizer = optimizer_list[train_config["optimizer"]](
            net.parameters(),
            lr = float(train_config["lr"]),
            momentum = float(train_config["momentum"]),
            weight_decay = float(train_config["weight_decay"])
        )
    else:
        optimizer = optimizer_list[train_config["optimizer"]](
            net.parameters(),
            lr = float(train_config["lr"]),
            weight_decay = float(train_config["weight_decay"])
        )
    if "lr_decay" in train_config and "lr_step_size" in train_config:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=int(train_config["lr_step_size"]), gamma=float(train_config["lr_decay"]))
    else:
        scheduler = None
    return net, optimizer, scheduler

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    if "train" not in config:
        raise Exception("Error: Cannot find the configuration of the training procedure in %s" % config_path)
    if "model" not in config:
        raise Exception("Error: Cannot find the configuration of the model in %s" % config_path)
    train_config = dict(config["train"])
    model_config = dict(config["model"])
    if train_config["loss"] == "mse":
        model_config["w_softmax"] = "yes"
    elif train_config["loss"] == "ce":
        model_config["w_softmax"] = "no"
    else:
        raise Exception("Error: Please define whether to add a softmax layer in the model and which function to use for computing loss")
    return train_config, model_config

def parse_args():
    parser = argparse.ArgumentParser(description="TrainModel")
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
        "--shuffle", "-s", dest="shuffle", type=bool, default=True, help="Whether to shuffle batches during training"
    )
    parser.add_argument(
        "--save_freq", "-f", dest="save_freq", type=int, default=50, help="Frequency of saving trained models"
    )
    parser.add_argument(
        "--model_ids", "-mi", dest="model_ids", type=str, default="", help="The ID of models for training. To specify the IDs, you have two choices. If you want to train the first, the second and the third models together, you can either provide '1,2,3' or '1~3'."
    )
    parser.add_argument(
        "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
    )
    args = parser.parse_args()
    return args

def log_args(args, out_dir):
    out_path = os.path.join(out_dir, "train_args.txt")
    in_str = []
    for key in sorted(list(args.keys())):
        in_str.append("%s: %s\n" % (str(key), str(args[key])))
    utils.write_txt(out_path, in_str)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # load configurations
    if not os.path.exists(args.config_file):
        raise Exception("Error: Cannot find the configuration file ... %s" % args.config_file)
    train_config, model_config = load_config(args.config_file)

    # check the criterion
    if train_config["loss"] == "mse":
        train_fn = train_mse
        test_fn = test_mse
    elif train_config["loss"] == "ce":
        train_fn = train_ce
        test_fn = test_ce
    else:
        raise Exception("Error: Unknown loss function ... %s" % train_config["loss"])
    if train_config["optimizer"] == "adam":
        if "momentum" in train_config:
            raise Exception("Error: No momentum when the optimizer is adam.")

    # load the indexes of samples in training subsets
    if not os.path.exists(args.idx_file):
        raise Exception("Error: Cannot find the index file for training subsets ... %s" % args.idx_file)
    idx_data = utils.read_npy(args.idx_file)
    if args.model_ids == "":
        model_id_list = range(1, idx_data["train_idx_sets"].shape[0]+1)
    elif "~" in args.model_ids:
        tmp = args.model_ids.split("~")
        model_id_list = list(range(int(tmp[0]), int(tmp[1])+1))
    else:
        model_id_list = [int(tmp) for tmp in args.model_ids.split(",")]
    print("Planning to the following models: %s" % str(model_id_list))
    out_dir = os.path.dirname(args.config_file)
    print("Saving models in %s" % out_dir)
    log_args(vars(args), out_dir)

    train_set, test_set = load_complete_data(args.data_tag, args.data_dir)
    test_loader = prepare_test_set(args.data_tag, test_set, idx_data["test_idx_set"], int(train_config["test_batch_size"]))
    # for model_id in range(1, model_num+1):
    for model_id in model_id_list:
        log_path = os.path.join(out_dir, "log_%d.txt" % model_id)
        utils.write_txt(log_path, ["Prepared for model-%d ...\n" % model_id])
        model_path = os.path.join(out_dir, "final_model_%d.pkl" % model_id)
        if os.path.exists(model_path):
            continue
        # prepare model
        net, optimizer, scheduler = init_model_and_train(model_config, train_config)
        # prepare dataset
        train_set, test_set = load_complete_data(args.data_tag, args.data_dir)
        train_loader = prepare_train_set(args.data_tag, train_set, idx_data["train_idx_sets"][model_id-1], int(train_config["train_batch_size"]), args.shuffle)

        train_set, test_set = load_complete_data(args.data_tag, args.data_dir)
        train_eval_loader = prepare_test_set(args.data_tag, train_set, idx_data["train_idx_sets"][model_id-1], int(train_config["test_batch_size"]))
        
        utils.write_txt(log_path, ["#Epoch\tTrainAcc\tTestAcc\tTrainLoss\tTestLoss\n"])
        for epoch_no in range(1, int(train_config["epoch_num"])+1):
            train_fn(train_loader, net, int(model_config["out_dim"]), optimizer)
            test_loss, test_acc = test_fn(test_loader, net, int(model_config["out_dim"]))
            train_loss, train_acc = test_fn(train_eval_loader, net, int(model_config["out_dim"]))
            utils.write_txt(log_path, ["%d\t%f\t%f\t%f\t%f\n" % (epoch_no, train_acc, test_acc, train_loss, test_loss)])
            if epoch_no % args.save_freq == 0:
                torch.save(
                    net.state_dict(), 
                    os.path.join(out_dir, "model_%d-epoch_%d.pkl" % (model_id, epoch_no))
                )
            if scheduler is not None:
                scheduler.step()
        torch.save(
            net.state_dict(), 
            model_path
        )
    print("Done!")

if __name__ == '__main__':
    main()
    
