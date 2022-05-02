import csv
import os
import sys
import pickle
import numpy as np

Attributes = ["Dataset","EpochNum","Arch","Width","TrainSize","Loss","Scheduler?","lr"]
EpochNums = [500,400]
Setups = {
    "cifar10": ["densenet161", "resnet34", "alexnetwol"],
    "cifar100": ["densenet161", "resnet34", "alexnetwol"],
    "mnist": ["mlp_1hidden"]
}
Archs = {
    "densenet161": [2,4,8,16,32],
    "resnet34": [2,4,8,16,32],
    "alexnetwol": [16,32,64,128,256],
    "mlp_1hidden": [8,16,32,64,128,256],
}
TrainSizes = ["1k", "5k"]
Losses = ["mse", "ce"]
Schedulers = ["with_scheduler", "without_scheduler"]
LRs = {
    "densenet161": 0.1,
    "resnet34": 0.1,
    "alexnetwol": 0.01,
    "mlp_1hidden": 0.1,
}

def read_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def main(target_dir, epsilon, output_dir):
    fieldnames = Attributes + [f"MemguardMemInfAcc(eps={epsilon})", f"MemguardNonmemInfAcc(eps={epsilon})", f"MemguardInfAcc(eps={epsilon},lmd=0.5)"]
    
    results = []
    for dataset in Setups:
        for epoch_num in EpochNums:
            for arch in Setups[dataset]:
                for width in Archs[arch]:
                    for train_size in TrainSizes:
                        for loss in Losses:
                            if loss == "mse" and dataset == "cifar100":
                                continue
                            for scheduler in Schedulers:
                                lr = LRs[arch]
                                tmp = f"{target_dir}/epoch_{epoch_num}/{scheduler}-{loss}/{dataset}-{arch}-w_{width}-ntrain_{train_size}-N_3"
                                if not os.path.isdir(tmp):
                                    tmp = f"{tmp}-less_lr"
                                if not os.path.isdir(tmp):
                                    raise Exception(f"ERROR: Cannot find the folder ... {tmp}")
                                memguard_dir = os.path.join(tmp, "memguard")
                                memguard_acc_dir = os.path.join(tmp, "memguard_acc")
                                item = [dataset, epoch_num, arch, width, train_size, loss, scheduler, lr, ]
                                # print(item, memguard_acc_dir)
                                if not os.path.isdir(memguard_acc_dir):
                                    item += ["-", "-", "-"]
                                else:
                                    filenames = os.listdir(memguard_acc_dir)
                                    mem_acc_list = []
                                    nonmem_acc_list = []
                                    acc_list = []
                                    for filename in filenames:
                                        filepath = os.path.join(memguard_acc_dir, filename)
                                        tmp2 = read_pkl(filepath)[epsilon]
                                        mem_acc_list.append(tmp2["MemberAcc"])
                                        nonmem_acc_list.append(tmp2["NonmemberAcc"])
                                        acc_list.append(0.5*tmp2["MemberAcc"] + 0.5*tmp2["NonmemberAcc"])
                                    mem_acc = np.mean(mem_acc_list)
                                    nonmem_acc = np.mean(nonmem_acc_list)
                                    acc = np.mean(acc_list)
                                    item += [mem_acc, nonmem_acc, acc]
                                results.append(item)
                                    
    output_path = os.path.join(output_dir, f"memguard-inf_acc-eps-{epsilon}.csv")
    with open(output_path, mode="w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(fieldnames)
        for item in results:
            writer.writerow(item)

if __name__ == "__main__":
    target_dir = sys.argv[1]
    epsilon = float(sys.argv[3])
    output_dir = sys.argv[2]
    main(target_dir, epsilon, output_dir)