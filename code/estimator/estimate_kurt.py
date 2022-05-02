import sys
sys.path.append("../")


import loader.utils as utils
import os
import torch
import loader.prepare_dataset as prep_data
import torch.nn as nn
import argparse
from trainer import models
from trainer import train
import torch.nn.functional as  function

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

def compute_bias_variance_mse(loss_collection, acc_collection, output_collection, norm_square_output_collection, model, testloader, class_num, model_num):
    # Set to be evaluation model
    model.eval()
    cnt = 0
    loss = 0
    correct_num = 0
    bias_square = 0
    variance = 0
    individual_loss_array = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
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
            individual_loss_array += list(torch.sum(raw_criterion_list["mse"](outputs, labels_onehot), dim=1).cpu().numpy())
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
    loss_collection = loss_collection + loss/cnt
    # percentage
    acc_collection = acc_collection + 100.0 * correct_num / cnt
    return individual_loss_array, loss_collection/model_num, acc_collection/model_num, bias_square/cnt, variance/cnt, output_collection, norm_square_output_collection, loss_collection, acc_collection

def calc_kl_div(p,q):
    return (p * (p / q).log()).sum()

def compute_bias_variance_ce(model, testloader, normalized_avg):
    model.eval()
    cnt = 0
    bias2 = 0
    variance = 0
    test_loss = 0
    correct_num = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            input_num = labels.size(0)

            outputs = model(inputs)

            outputs = outputs.type(torch.DoubleTensor).cuda()
            # outputs = outputs.cuda()

            batch_loss = criterion_list["ce"](outputs, labels)
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
    return test_loss/cnt, 100.0* correct_num/cnt, bias2/cnt, variance/cnt

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

def log_mse(log_path, repeat_num, model_num, loss, acc, variance, bias_square, unbiased_variance, unbiased_bias_square):
    log_str = "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f" % (
        repeat_num, model_num, loss, acc, variance, bias_square, unbiased_variance, unbiased_bias_square
    )
    print(log_str)
    utils.write_txt(log_path, [log_str+"\n"])

def log_ce(log_path, repeat_num, model_num, loss, acc, variance, bias_square):
    log_str = "%d\t%d\t%f\t%f\t%f\t%f" % (
        repeat_num, model_num, loss, acc, variance, bias_square
    )
    print(log_str)
    utils.write_txt(log_path, [log_str+"\n"])

def estimate_ce(log_path, model_config, test_loader, model_paths_dict, repeat_id,normalized_avg):
    variance_sum = 0.0
    for file_id in model_paths_dict:
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        model.load_state_dict(torch.load(model_paths_dict[file_id]))
        loss, acc, bias2, variance = compute_bias_variance_ce(model, test_loader, normalized_avg)
        variance_sum += variance
        variance_avg = variance_sum / float(file_id)
        log_ce(log_path, repeat_id, file_id, loss, acc, variance_avg, bias2)

def estimate_mse(log_path, model_config, test_size, test_loader, model_paths_dict, repeat_id):
    class_num = int(model_config["out_dim"])
    output_collection = torch.Tensor(test_size, class_num).zero_().cuda()
    norm_square_output_collection = torch.Tensor(test_size).zero_().cuda()
    loss_collection = 0
    acc_collection = 0
    # model_num = len(model_paths)
    detailed_losses = {}
    detailed_loss_path = os.path.join(os.path.dirname(log_path), "detailed_loss.pkl")
    for file_id in model_paths_dict:
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        model.load_state_dict(torch.load(model_paths_dict[file_id]))
        individual_losses, loss, acc, bias_square, variance, output_collection, norm_square_output_collection, loss_collection, acc_collection = compute_bias_variance_mse(loss_collection, acc_collection, output_collection, norm_square_output_collection, model, test_loader, class_num, file_id)
        detailed_losses[file_id] = individual_losses
        unbiased_variance = variance * file_id / (file_id - 1.0)
        unbiased_bias_square = loss - unbiased_variance
        log_mse(log_path, repeat_id, file_id, loss, acc, variance, bias_square, unbiased_variance, unbiased_bias_square)
    utils.write_pkl(detailed_loss_path, detailed_losses)

def compute_kurt_ce(log_path, model_config, test_loader, model_paths_dict, repeat_id, normalized_avg,
                    compute_over_train=None):
    detailed_losses = {}
    detailed_var = {}
    model_num = len(model_paths_dict)
    variance_sum = 0.0

    for model_id in model_paths_dict:
        model = models.MODELS[model_config["model_tag"]](model_config).cuda()
        model.load_state_dict(torch.load(model_paths_dict[model_id]))
        if compute_over_train:
            train_loader = compute_over_train(model_id)
            loss, acc, bias2, variance = compute_bias_variance_ce(model, train_loader, normalized_avg)
        else:
            loss, acc, bias2, variance = compute_bias_variance_ce(model, test_loader, normalized_avg)
        detailed_losses[model_id] = loss
        variance_sum += variance
        variance_avg = variance_sum / float(model_id)
        log_ce(log_path, repeat_id, model_id, loss, acc, variance_avg, bias2)

    kurtosis = 0
    skewness = 0
    print("avg loss: {}; avg variance: {}".format(loss, variance_avg))
    print("averaged over {}".format(model_num))
    for model_id in range(1, model_num+1):
        model_loss = detailed_losses[model_id]
        kurtosis += ((model_loss - loss)**4 / variance_avg**2)
        skewness += ((model_loss - loss)**3 / variance_avg**(3/2))

    print("kurtosis: {}; skewness: {}".format(kurtosis, skewness))
    print("kurtosis avg: {}; skewness avg: {}".format(kurtosis / model_num, skewness / model_num))

    return kurtosis / model_num, skewness / model_num, variance_avg, loss

def precomp_kurt(log_path, testloader):
    print(log_path)
    f = open(log_path)
    for line in f:
        data = line.split('\t')
        loss = float(data[2])
        variance = float(data[4])
    f.close()
    print(loss, variance)
#    model.eval()
#    cnt = 0
#    test_loss = 0
#    correct_num = 0
#    with torch.no_grad():
#        for idx, (inputs, labels) in enumerate(testloader):
#            inputs = inputs.cuda()
#            labels = labels.long().cuda()
#            input_num = labels.size(0)
#
#            outputs = model(inputs)
#
#            outputs = outputs.type(torch.DoubleTensor).cuda()
#
#            batch_loss = criterion_list["ce"](outputs, labels)
#            test_loss += batch_loss.item() * input_num
#            _, preds = outputs.max(1)
#            correct_num += preds.eq(labels).sum().item()
#
#    test_loss = test_loss/cnt
#    correct_num = 100.0* correct_num/cnt


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
        "--model_num_per_repeat", "-n", dest="model_num_per_repeat", type=int, default=6, help="Number of models to train per trial"
    )
    parser.add_argument(
        "--gpu_id", "-g", dest="gpu_id", type=str, default="0", required=False, help="The ID of the GPU to use"
    )
    parser.add_argument(
        "--kurt", default=False, required=False, action="store_true", help="Compute the kurtosis"
    )
    parser.add_argument(
        "--precomp_kurt", default=False, required=False, action="store_true", help="Compute kurtosis based on precomputed bias-variance"
    )
    parser.add_argument(
        "--train", default=False, required=False, action="store_true", help="Compute over training"
    )
    parser.add_argument(
        "--use_first_n_models_per_split", "-um", default=10,  type=int, required=False,  help="if it is 10, then 1~10, 51~60... will be used for estimation"
    )
    parser.add_argument(
        "--number_of_split", "-ns", default=3,  type=int, required=False,  help="N=3"
    )
    args = parser.parse_args()
    return args

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

def get_model_paths_dicts(model_dir, model_num_per_repeat, use_first_n_models_per_split=10, N=3):
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
        for model_id in range(use_first_n_models_per_split):
            file_id = model_id + round_id * model_num_per_repeat + 1
            if file_id not in file_dict:
                print('skipping it for now... file_id {} missing'.format(file_id))
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

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # load config
    if not os.path.exists(args.config_file):
        raise Exception("Error: Configuration file does not exist ... %s" % args.config_file)
    train_config, model_config = train.load_config(args.config_file)

    # load data
    if not os.path.exists(args.idx_file):
        raise Exception("Error: Data idx file does not exist ... %s" % args.idx_file)
    train_set, test_set = train.load_complete_data(args.data_tag, args.data_dir)
    test_size = test_set.data.shape[0]
    # load idx data
    idx_data = utils.read_npy(args.idx_file)

    model_dir = os.path.dirname(args.config_file)
    log_args(vars(args), model_dir)

    log_path = os.path.join(model_dir, "bv.txt")
    if args.precomp_kurt:
        print("Use the already computed variance and mean to estimate the kurtosis")
        precomp_kurt(log_path)
        return
    if os.path.exists(log_path):
        os.remove(log_path)


    # prepare test loader
    test_loader = train.prepare_test_set(args.data_tag, test_set, idx_data["test_idx_set"],
                                             int(train_config["test_batch_size"]))


    model_path_array_of_dicts= get_model_paths_dicts(model_dir, args.model_num_per_repeat, args.use_first_n_models_per_split, args.number_of_split)
    print("The path dict is:", model_path_array_of_dicts)
    if train_config["loss"] == "ce":
        # compute log-output average
        #logprobs_sum = torch.Tensor(test_size, int(model_config["out_dim"])).zero_().cuda()
        logprobs_sum = torch.Tensor(test_size, int(model_config["out_dim"])).zero_().type(torch.DoubleTensor).cuda()
        model_cnt = 0
        for model_paths_dict in model_path_array_of_dicts:
            for file_id in model_paths_dict:
                model_path = model_paths_dict[file_id]
                model_cnt += 1
                model = models.MODELS[model_config["model_tag"]](model_config).cuda()
                model.load_state_dict(torch.load(model_path))
                logprobs_sum += compute_log_output(model, test_loader, int(model_config["out_dim"]), test_size)
        logprobs_avg = logprobs_sum / float(model_cnt)
        normalized_avg = compute_normalized_kl(logprobs_avg, int(model_config["out_dim"]), test_size)
    # compute the bias2 & variance
    repeat_id = 0
    if train_config["loss"] == "mse":
        log_str = "#repeat\t#model\ttest loss\ttest acc\tvariance\tbias2\tunbiased variance\tunbiased bias2"
    elif train_config["loss"] == "ce":
        log_str = "#repeat\t#model\ttest loss\ttest acc\tvariance\tbias2"
    else:
        raise Exception("Error: The log format is not defined")
    print(log_str)
    utils.write_txt(log_path, [log_str + "\n"])

    skew = 0
    kurtosis = 0
    variance_avg = 0
    bias_avg = 0
    for model_paths_dict in model_path_array_of_dicts:
        repeat_id += 1
        if args.kurt:
            if args.train:
                # the prepare_train_set modifies the train_set
                print('Compute over training data')
                compute_over_train = lambda model_id : train.prepare_train_set(args.data_tag,
                                                                               train.load_complete_data(args.data_tag,
                                                                                                        args.data_dir)[0],
                                                                               idx_data["train_idx_sets"][model_id - 1],
                                                                               int(train_config["train_batch_size"]),
                                                                               shuffle_tag=False)
            else:
                compute_over_train = None

            kurtosis_round, skew_round, var_round, bias_round = compute_kurt_ce(log_path, model_config, test_loader, model_paths_dict, repeat_id,
                                                                                normalized_avg, compute_over_train)
            kurtosis += kurtosis_round
            skew += skew_round
            variance_avg += var_round
            bias_avg += bias_round
        elif train_config["loss"] == "mse":
            estimate_mse(log_path, model_config, test_size, test_loader, model_paths_dict, repeat_id)
        elif train_config["loss"] == "ce":
            estimate_ce(log_path, model_config, test_loader, model_paths_dict, repeat_id, normalized_avg)
        else:
            raise Exception("Error: The estimation function is not defined!")
    if args.kurt:
        print("final kurtosis: {}; final skew: {}".format(kurtosis / len(model_paths_dict),
                                                          skew / len(model_paths_dict)))
        print("final bias: {}; final variance: {}".format(bias_avg / len(model_paths_dict),
                                                          variance_avg / len(model_paths_dict)))

        print("Done!")



if __name__ == '__main__':
    main()
    
