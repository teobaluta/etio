import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import train


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

def parse_log_file(file_path, tags):
    with open(file_path) as f:
        lines = f.readlines()
    tmp = lines[0].strip().split(",")
    tmp2 = lines[1].strip().split(",")
    data = []
    for tag in tags:
        data.append(tmp2[tmp.index(tag)])
    return data

def main(base_dir, out_dir):
    fieldnames = [
        "Dataset", "Arch", "Width", "TrainSize", "Loss", 
        "Scheduler?", "lr", "EpochNum", "WeightDecay"
    ]
    tags = ["sorted_3", "sorted_10", "origin"]
    fieldnames += [f"CentroidDistance({tag})" for tag in tags]

    stats = []
    for epoch_num in EpochNums:
        epoch_dir_path = os.path.join(base_dir, "epoch_%d" % epoch_num)
        for data in DataDict:
            for ntrain in TrainSizeList:
                for loss in LossDict:
                    if data not in LossDict[loss]:
                        continue
                    for scheduler in SchedulerList:
                        for arch in DataDict[data]:
                            lr = "0.1"
                            if arch == "alexnetwol":
                                lr = "0.01"
                            for width in WidthDict[arch]:
                                setup_path = os.path.join(
                                    epoch_dir_path,
                                    "%s-%s/%s-%s-w_%d-ntrain_%s-N_3%s" % (
                                        scheduler, loss,
                                        data, arch, width, ntrain, "-less_lr"
                                    )
                                )
                                file_path = os.path.join(
                                    setup_path,
                                    "centroid_distance.txt"
                                )
                                if not os.path.exists(file_path):
                                    setup_path = os.path.join(
                                        epoch_dir_path,
                                        "%s-%s/%s-%s-w_%d-ntrain_%s-N_3%s" % (
                                            scheduler, loss,
                                            data, arch, width, ntrain, ""
                                        )
                                    )
                                    file_path = os.path.join(
                                        setup_path,
                                        "centroid_distance.txt"
                                    )
                                config_file = os.path.join(setup_path, "config.ini")
                                # load configuration
                                if os.path.exists(config_file):
                                    train_config, _ = train.load_config(config_file)
                                    config_epoch_num = train_config["epoch_num"]
                                    weight_decay = train_config["weight_decay"]
                                    assert (int(config_epoch_num) == int(epoch_num)), f"epoch_num {epoch_num} not matching with config {config_epoch_num}"
                                    item = [data, arch, width, ntrain, loss, scheduler, lr, epoch_num, weight_decay]
                                else:
                                    print(f"WARNING! Missing config file {config_file}")
                                    item = [data, arch, width, ntrain, loss, scheduler, lr, epoch_num, "-"]

                                if os.path.exists(file_path):
                                    item += parse_log_file(file_path, tags)
                                else:
                                    item += ["-", "-", "-"]
                                stats.append(item)
    out_path = os.path.join(out_dir, "stats-centroid_distance.csv")
    with open(out_path, mode="w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(fieldnames)
        for item in stats:
            writer.writerow(item)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
