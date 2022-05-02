import os
import pdb
import sys
import parse_mlleaks
import parse_oak17
from itertools import product
import csv
import copy
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from estimator import estimate
import loader.utils as utils
from trainer import train
from estimator.gather_centroid_distance import parse_log_file
from estimator.count_params import get_param_num
from estimator import get_centroid
from defence import eval_memguard
from estimator import estimate_memguard

REPEAT_NUM = 3
WeightDecayMap = {
    "5e3": 5e-3,
    "5e4": 5e-4,
}
FORCE_STAT_REFRESH = False  # set this tag to True force all the stats to be recomputed: note that this may take lot of time

EpochNumsList = [500, 400]
SchedulerList = ["with_scheduler", "without_scheduler"]
LossDict = {
    "ce":  ["cifar10", "cifar100", "mnist"],
    "mse": ["cifar10", "mnist"]
}
DataDict = {
    "cifar10":  ["resnet34", "densenet161", "alexnetwol"],
    "cifar100": ["resnet34", "densenet161", "alexnetwol"],
    "mnist":    ["mlp_1hidden"]
}
ArchToWidthDict = {
    "resnet34":   [32, 16, 8, 4, 2],
    "densenet161":[32, 16, 8, 4, 2],
    "alexnetwol": [256, 128, 64, 32, 16],
    "mlp_1hidden":[256, 128, 64, 32, 16]
}
TrainSizeList = ["5k", "1k"]
ModelNumPerRepeat = {
    "cifar10": {"1k": 50, "5k": 10},
    "cifar100": {"1k": 50, "5k": 10},
    "mnist": {"1k": 60, "5k": 12},
}
TrainSizeMap = {"1k": "1000",  "5k": "5000", "10k": "10000"}


# Following dictionaries are used in parsing mlleaks and oak17 attack respectively
SchedulerMaps = {
    "with_scheduler": "w_scheduler",
    "without_scheduler": "wo_scheduler"
}
Oak17ShadowIDMap = {
    "5e3": 
        {
            "with_scheduler": {
                "1k": {"cifar10": "2~10", "cifar100": "2~10", "mnist": "2~60"},
                "5k": {"cifar10": "2~10", "cifar100": "2~10", "mnist": "2~12"}
            },
            "without_scheduler": {
                "1k":  {"cifar10": "2~50", "cifar100": "2~50", "mnist": "2~60"},
                "5k":  {"cifar10": "2~10", "cifar100": "2~10", "mnist": "2~12"}
            }
        },
    "5e4": 
        {
            "with_scheduler": {
                "1k":  {"cifar10": "2~50", "cifar100": "2~50", "mnist": "2~60"},
                "5k":  {"cifar10": "2~10", "cifar100": "2~10", "mnist": "2~12"}
            },
            "without_scheduler": {
                "1k":  {"cifar10": "2~50", "cifar100": "2~50", "mnist": "2~60"},
                "5k":  {"cifar10": "2~10", "cifar100": "2~10", "mnist": "2~12"}
            }
        },
}

# maps the field name from the summary files of oak17 attack to the name we use here
oak17_attack_fields_map = {
    "AttackTestAcc": "Oak17MemAcc",   
    "AttackTrainAcc": "Oak17NonmemAcc",
    "Attack(0.5)": "Oak17Acc",
    }

ModelIdentityFields= ["WeightDecay", "EpochNum", "Dataset", "Arch", 
"Width", "TrainSize", "Loss", "Scheduler?","lr"]
MlleaksFields = ["MLLeakMemAcc", "MLLeakNonmemAcc", "MLLeakAcc", "MLLeakLMemAcc",
 "MLLeakLNonmemAcc", "MLLeakLAcc"]
MlleakAttacksDict = {
    "mlleaks-attacks": "",
    "mlleaks_top3-attacks": "Top3"
}

Oak17Fields = ["Oak17MemAcc", "Oak17NonmemAcc", "Oak17Acc"]
ComputeTagsName = ["CentroidDistance(sorted_3)", "CentroidDistance(sorted_10)", "CentroidDistance(origin)"]
ComputeTags = ["sorted_3", "sorted_10", "origin"]
StatHeader = ["TrainAcc", "TestAcc", "TrainLoss", "TestLoss", 
"TrainVar", "TestVar", "TrainBias2", "TestBias2", "AccDiff", "LossDiff"]

MemguardStatHeader = ["MemguardTrainAcc", "MemguardTestAcc", "MemguardTrainLoss", "MemguardTestLoss",
"MemguardTestVar", "MemguardTestBias2", "MemguardAccDiff", "MemguardLossDiff"]

NumParams = ["NumParams"]
#memguard_stat_fieldnames = ["MemguardTrainAcc", "MemguardTestAcc", "MemguardTrainLoss", "MemguardTestLoss", 
#"MemguardTrainVar", "MemguardTestVar", "MemguardTrainBias2", "MemguardTestBias2", "MemguardAccDiff", "MemguardLossDiff"]

MemguardAccFields = ["MemguardMemberAcc", "MemguardNonmemberAcc", "MemguardAcc"]

ThresholdField = ["ThresholdMemAcc", "ThresholdNonMemAcc", "ThresholdAcc", "Threshold"]
RECORD_FIELDS = []
RECORD_FIELDS += ModelIdentityFields
RECORD_FIELDS += NumParams
RECORD_FIELDS += StatHeader
RECORD_FIELDS += ComputeTagsName

for mlleak_attack in MlleakAttacksDict:
    fieldname_prefix = MlleakAttacksDict[mlleak_attack]
    for f in MlleaksFields:
        RECORD_FIELDS.append(fieldname_prefix + f)
RECORD_FIELDS += Oak17Fields
RECORD_FIELDS += ThresholdField
RECORD_FIELDS += MemguardStatHeader
RECORD_FIELDS += MemguardAccFields
MISSING_RECORD_HEADER = ["AttackName"] + ModelIdentityFields

# Defining Helper Functions

def create_record_from_dict(setup_dict , fieldnames):
    """
    Returns a list in same order as fields in fieldnames with values from setup_dict
    If the field does not exists, it replaces it with "-"
    """
    record = []
    for field in fieldnames:
        if field in setup_dict:
            record.append(setup_dict[field])
        else:
            record.append("-")
    return record

def get_setup_dict(setup, fields):
    setup_dict = {}
    for i in range(len(fields)):
        field = fields[i]
        setup_dict[field] = setup[i]
    return setup_dict


def parse_mlleak_csv_file(file_path):
    data = []
    attrs = None
    tag = False
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if tag:
                data.append(row)
            if "Avg Train Attack Acc" in row:
                attrs = row
                tag = True
    w_idx = attrs.index("Width")
    w_label_idx = attrs.index("w_label")
    mem_attack_acc_idx = attrs.index('Avg Mem Attack Acc')
    nonmem_attack_acc_idx = attrs.index('Avg Non-mem Attack Acc')
    overall_attack_cc_idx = attrs.index('Avg Overall Attack Acc')
    data_dict = {}
    for row in data:
        width = int(row[w_idx].split("_")[1])
        if width not in data_dict:
            data_dict[width] = {}
        if row[w_label_idx] == "True":
            tag = "L"
        else:
            tag = ""
        data_dict[width]["MLLeak%sMemAcc" % tag] = row[mem_attack_acc_idx]
        data_dict[width]["MLLeak%sNonmemAcc" % tag] = row[nonmem_attack_acc_idx]
        data_dict[width]["MLLeak%sAcc" % tag] = row[overall_attack_cc_idx]
    return data_dict


# Handler for MLLeak Attack Summaries

def handle_mlleaks_attack(base_dir, attack_name, fieldname_prefix, width_to_setup_dict, setup_dict, missing_records):

    mlleak_attack_log_name = "{dataset}-{arch}-{tsize}-{w_sch}-{loss}".format(
        dataset=setup_dict["Dataset"],
        arch=setup_dict["Arch"],
        tsize=setup_dict["TrainSize"],
        w_sch=SchedulerMaps[setup_dict["Scheduler?"]],
        loss=setup_dict["Loss"])

    attack_dir = os.path.join(
        base_dir,
        attack_name,
        "{attack_name}-wd_{wd}".format(attack_name = attack_name, wd = setup_dict["WeightDecay"]),
        "epoch_{epoch_num}".format(epoch_num = setup_dict["EpochNum"]),
        "logs",
        mlleak_attack_log_name)

    if not os.path.exists(attack_dir):
        setup_dict["AttackName"] = attack_name
        missing_records.append(create_record_from_dict(setup_dict, MISSING_RECORD_HEADER))
    else:
        print("Parsing MLLeak attack from {}".format(attack_dir))
        attack_summary_file_path = os.path.join(
            summary_dir,
            attack_name,
            "{mlleak_log_name}-epoch_{epoch_num}-wd_{wd}.csv".format(
                mlleak_log_name = mlleak_attack_log_name,
                epoch_num = setup_dict["EpochNum"],
                wd = setup_dict["WeightDecay"])
            )

        parse_mlleaks.main(attack_dir,attack_summary_file_path)
        ml_leak_dict = parse_mlleak_csv_file(attack_summary_file_path)
        for width in ArchToWidthDict[setup_dict["Arch"]]:
            curr_setup_dict = width_to_setup_dict[width]
            if width not in ml_leak_dict:
                curr_setup_dict["AttackName"] = attack_name
                missing_records.append(create_record_from_dict(curr_setup_dict, MISSING_RECORD_HEADER))
            else:
                curr_mlleak_dict = ml_leak_dict[width]
                for field in curr_mlleak_dict:
                    curr_setup_dict[fieldname_prefix + field] = curr_mlleak_dict[field]

# Handler for Oak17 Attack Summary
def handle_oak17_attack(base_dir, width_to_setup_dict, missing_records):
    for width in width_to_setup_dict:
        setup_dict = width_to_setup_dict[width]
        setup_dict["Width"] = width
        model_name = setup_dict["ModelName"]

        # /srv/home/hitarth/teogang/oak17-attacks/oak17-attacks-wd_5e3/epoch_500/without_scheduler-ce/shadow_attack_cifar100-densenet161-w_16-ntrain_1k-N_3-target_1-shadow_2~50-seeds_0,1
        shadow_attack_dir_name = "shadow_attack_{model_name}-target_{target}-shadow_{shadowids}-seeds_{seeds}".format(
            model_name=model_name,
            target=1,
            shadowids=Oak17ShadowIDMap[setup_dict["WeightDecay"]][setup_dict["Scheduler?"]][setup_dict["TrainSize"]][setup_dict["Dataset"]],
            seeds="0,1")

        shadow_attack_summary_path = os.path.join(
            base_dir,
            "oak17-attacks",
            "oak17-attacks-wd_{wd}".format(wd=setup_dict["WeightDecay"]),
            "epoch_{epoch_num}".format(epoch_num=setup_dict["EpochNum"]),
            "{w_sch}-{loss}".format(w_sch=setup_dict["Scheduler?"], loss=setup_dict["Loss"]),
            shadow_attack_dir_name,
            "shadow_attack_summary.csv")

        if not os.path.exists(shadow_attack_summary_path):
            setup_dict["AttackName"] = "oak17-attacks"
            missing_records.append(create_record_from_dict(setup_dict, MISSING_RECORD_HEADER))
        else:
            print("Parsing Oak17 attack from {}".format(shadow_attack_summary_path))
            oak17_dict = parse_oak17.get_oak17_summary_dict_for_model(shadow_attack_summary_path)
            if oak17_dict == {}:
                setup_dict["AttackName"] = "oak17-attacks"
                missing_records.append(create_record_from_dict(setup_dict, MISSING_RECORD_HEADER)) 
            else:
                for field in oak17_attack_fields_map:
                    setup_dict[oak17_attack_fields_map[field]] = oak17_dict[field]/100 if field in oak17_dict else "-"

def get_average_or_negative(lst):
    if len(lst) == 0:
        return -1
    return sum(lst)/len(lst)


def handle_threshold_attack(base_dir, width_to_setup_dict, missing_records):
    for width in width_to_setup_dict:
        setup_dict = width_to_setup_dict[width]
        setup_dict["Width"] = width
        model_name = setup_dict["ModelName"]
        threshold_dict_path = os.path.join(
                    base_dir,
                    "models",
                    "models-wd_{wd}".format(wd=setup_dict["WeightDecay"]),
                    "epoch_{epoch_num}".format(epoch_num=setup_dict["EpochNum"]),
                    "{w_sch}-{loss}".format(w_sch=setup_dict["Scheduler?"], loss=setup_dict["Loss"]),
                    model_name,
                    "results-threshold_csf.pkl")
        if not os.path.exists(threshold_dict_path):
            print("Warning: Missing threshold attack file: %s"%threshold_dict_path)
            setup_dict["AttackName"] = "ThresholdAttack"
            missing_records.append(create_record_from_dict(setup_dict, MISSING_RECORD_HEADER))
        else:
            acc_dict = utils.read_pkl(threshold_dict_path)
            member_accuracy_lst = []
            nonmember_accuracy_lst = []

            for i in range(1, REPEAT_NUM+1):
                # for each repeat id, get the average value of threshold attack for all models in the current `repeat_id` split
                repeat_id = i
                thresh_dict = acc_dict["repeat_%d" % repeat_id]
                if len(thresh_dict["train"]) == 0 or len(thresh_dict["test"]) == 0:
                    print("[Threshold Attack Parse] Ignoring repeat: ", repeat_id)
                    continue
                member_accuracy_lst.append(get_average_or_negative(thresh_dict["train"]))
                nonmember_accuracy_lst.append(get_average_or_negative(thresh_dict["test"]))
            
            member_accuracy = get_average_or_negative(member_accuracy_lst)
            nonmember_accuracy = get_average_or_negative(nonmember_accuracy_lst)

            if member_accuracy <= 0 or nonmember_accuracy <=0:
                return 
            threshold = acc_dict["threshold"] 

            setup_dict["ThresholdMemAcc"] = member_accuracy
            setup_dict["ThresholdNonMemAcc"] = nonmember_accuracy
            setup_dict["ThresholdAcc"] = (member_accuracy + nonmember_accuracy)/2
            setup_dict["Threshold"] = threshold


def handle_model_stats(base_dir, setup_dict):
    model_name = setup_dict["ModelName"]
    setup_dir_path = os.path.join(
                base_dir,
                "models",
                "models-wd_{wd}".format(wd=setup_dict["WeightDecay"]),
                "epoch_{epoch_num}".format(epoch_num=setup_dict["EpochNum"]),
                "{w_sch}-{loss}".format(w_sch=setup_dict["Scheduler?"], loss=setup_dict["Loss"]),
                model_name)

    full_setup_name = "%s-%s-%s-%s-%s"%(model_name, setup_dict["Scheduler?"], setup_dict["Loss"], setup_dict["EpochNum"], setup_dict["WeightDecay"])
    datasets_dir_path = os.path.join(base_dir, "datasets")
    if os.path.isdir(setup_dir_path):
        tag, stats = estimate.analyze_each_setup(datasets_dir_path, setup_dir_path, FORCE_STAT_REFRESH)
    else:
        print("Warning: Cannot find the folder ... %s" % setup_dir_path)
        tag = False
        return

    if tag and len(stats) == 11:
        for i in range(len(StatHeader)):
            field = StatHeader[i]
            setup_dict[field] = stats[i]
        detailed_losses = stats[-1]
        if detailed_losses:
            loss_dict[full_setup_name] = detailed_losses
            utils.write_pkl(loss_file, loss_dict)
    # for num_params
    handle_num_params(base_dir, setup_dict)
    # for centroid
    datasets_dir_path = os.path.join(base_dir, "datasets")
    data_idx_path = os.path.join(
        datasets_dir_path,
        "prepare_data-data_{dataset}-ntrial_3-train_size_{trainsize_number}-test_size_10000-strategy_disjoint_split-seed_0.npz".format(
            dataset = setup_dict["Dataset"], 
            trainsize_number = TrainSizeMap[setup_dict["TrainSize"]]))

    if not os.path.isdir(datasets_dir_path):
        print("Datasets not found at: ", datasets_dir_path)

    centroid_file_path = os.path.join(
        setup_dir_path,
        "centroid_distance.txt"
    )
    record = []
    if os.path.exists(centroid_file_path):
        print("Centroid stats exist {}. Parsing from there.".format(centroid_file_path))
        record = parse_log_file(centroid_file_path, ComputeTags)
        for i in range(len(ComputeTags)):
            tag = ComputeTags[i]
            setup_dict["CentroidDistance({})".format(tag)] = record[i]

    else:
        # Here, we should invoke the get_centroid computation if the
        # centroid_distance.txt does not exist.
        #! TODO Need to change the parsing to be threaded

        distance_list = []
        for compute_tag in ComputeTags:
            distance = get_centroid.compute_centroid_for_each_setup(
                setup_dir_path,
                setup_dict["Dataset"],
                datasets_dir_path,
                data_idx_path,
                ModelNumPerRepeat[setup_dict["Dataset"]][setup_dict["TrainSize"]], 
                setup_dir_path,
                full_setup_name,
                compute_tag
            )
            distance_list.append(str(distance))
            setup_dict["CentroidDistance({})".format(tag)] = str(distance)
        utils.write_txt(
            os.path.join(setup_dir_path, "centroid_distance.txt"), 
            [",".join(ComputeTags) + "\n", ",".join(distance_list) + "\n"]
        )


def handle_num_params(base_dir, setup_dict):
    model_name = setup_dict["ModelName"]
    setup_dir_path = os.path.join(
                base_dir,
                "models",
                "models-wd_{wd}".format(wd=setup_dict["WeightDecay"]),
                "epoch_{epoch_num}".format(epoch_num=setup_dict["EpochNum"]),
                "{w_sch}-{loss}".format(w_sch=setup_dict["Scheduler?"], loss=setup_dict["Loss"]),
                model_name)
    
    if not os.path.isdir(setup_dir_path):
        return

    num_param_out_path = os.path.join(
        setup_dir_path,
        "num_params.txt")

    if os.path.exists(num_param_out_path):
        print("Got num_params from file.")
        param_num = utils.read_txt(num_param_out_path)[0]
    else:
        model_path = os.path.join(setup_dir_path, "models/final_model_1.pkl")
        param_num = get_param_num(model_path, model_config)
        utils.write_txt(num_param_out_path, str(param_num))
    setup_dict["NumParams"] = param_num

# for memguard
def handle_memguard_acc(base_dir, setup_dict, missing_records):
    """
    Add memguard data to setup_dict
    """
    model_name = setup_dict["ModelName"]
    setup_dir_path = os.path.join(
                base_dir,
                "models",
                "models-wd_{wd}".format(wd=setup_dict["WeightDecay"]),
                "epoch_{epoch_num}".format(epoch_num=setup_dict["EpochNum"]),
                "{w_sch}-{loss}".format(w_sch=setup_dict["Scheduler?"], loss=setup_dict["Loss"]),
                model_name)
    print("Computing stats for Memguard...")
    full_setup_name = "%s-%s-%s-%s-%s"%(model_name, setup_dict["Scheduler?"], setup_dict["Loss"], setup_dict["EpochNum"], setup_dict["WeightDecay"])
    datasets_dir_path = os.path.join(base_dir, "datasets")

    memguard_base = os.path.join(setup_dir_path, "memguard")
    if not os.path.exists(memguard_base):
        setup_dict["AttackName"] = "memguard"
        missing_records.append(create_record_from_dict(setup_dict, missing_record_header))
        return

    member_acc, nonmember_acc = [], []
    EPSILON = 0.5 # Choosing one epsilon value: refer eval_memguard.py

    defence_files = os.listdir(memguard_base)
    for defence_file in defence_files:
        memguard_file = os.path.join(memguard_base, defence_file)
        mlleak_attack_model_name = "{dataset}-{arch}-{tsize}-{w_sch}-{loss}.pth".format(
        dataset=setup_dict["Dataset"],
        arch=setup_dict["Arch"],
        tsize=setup_dict["TrainSize"],
        w_sch=SchedulerMaps[setup_dict["Scheduler?"]],
        loss=setup_dict["Loss"])
        attack_name = "mlleaks_top3-attacks"

        attack_model_path = os.path.join(
            base_dir,
            attack_name,
            "{attack_name}-wd_{wd}".format(attack_name = attack_name, wd = setup_dict["WeightDecay"]),
            "epoch_{epoch_num}".format(epoch_num = setup_dict["EpochNum"]),
            "attack_models",
            mlleak_attack_model_name)

        acc_dict = eval_memguard.eval_memguard(memguard_file, attack_model_path, setup_dict["Arch"], setup_dict["Loss"])
        member_acc.append(acc_dict[EPSILON]["MemberAcc"])
        nonmember_acc.append(acc_dict[EPSILON]["NonmemberAcc"])

    if len(member_acc) == 0:
        print("Error: No memguard data found for model!")
        setup_dict["AttackName"] = "memguard"
        missing_records.append(create_record_from_dict(setup_dict, missing_record_header))
        return

    setup_dict["MemguardMemberAcc"] = np.mean(np.asarray(member_acc))
    setup_dict["MemguardNonmemberAcc"] =  np.mean(np.asarray(nonmember_acc))
    setup_dict["MemguardAcc"] = (setup_dict["MemguardMemberAcc"] + setup_dict["MemguardNonmemberAcc"])/2


    tag, stats = estimate_memguard.analyze_each_setup_memguard(datasets_dir_path, setup_dir_path, False, NO_EVALUATION = True)
    if tag and len(stats) == len(MemguardStatHeader)+1:
        for i in range(len(MemguardStatHeader)):
            field = MemguardStatHeader[i]
            setup_dict[field] = stats[i]
        detailed_losses = stats[-1]
        if detailed_losses:
            memguard_loss_dict[full_setup_name] = detailed_losses
            utils.write_pkl(memguard_loss_file, loss_dict)


def get_model_id_from_defence_file_name(defence_file):
    # memguard_file = "/mnt/archive0/shiqi/epoch_500/with_scheduler-mse/cifar10-densenet161-w_2-ntrain_1k-N_3/memguard/cifar10-densenet161-w_2-1k-target_model_id_14-shadow_id_33-nm_model_id_9-ml_leaks-shadow-loss_mse-attack_model.npz"
    lst = defence_file.split("-")
    for k in lst:
        if k.find("target_model_id") != -1:
            return int(k.split("_")[-1])
    return -1
        

def parse_all_attack_acc(base_dir, summary_dir):
    """
    Input: base_dir: The directory is assumed to have following structures:
    ├─────────────────────────────────── 
    │base_dir
    │── datasets
    │── mlleaks-attacks
    │   ├── mlleaks-attacks-wd_5e3
    │   │   ├── epoch_400
    │   │   └── epoch_500
    │   └── mlleaks-attacks-wd_5e4
    │   │   ├── epoch_400
    │   │   └── epoch_500
    ├── mlleaks_top3-attacks
    │   ├── mlleaks_top3-attacks-wd_5e3 
    │   │   ├── epoch_400
    │   │   ├── epoch_500 
    │   └── mlleaks_top3-attacks-wd_5e4 
    │       ├── epoch_400 
    │       └── epoch_500
    ├── oak17-attacks/
    │   ├── oak17-attacks-wd_5e3
    │   │   ├── epoch_400
    │   │   └── epoch_500
    │   └── oak17-attacks-wd_5e4
    │       ├── epoch_400
    │       └── epoch_500
    ├───────────────────────────────────
    Results:
    1. Creates summary files for all MLLeaks attacks in directory `summary/mlleaks*-attacks`
    2. Creates summary file `summary/full-summary.csv` for all the attacks
    3. Creates a log file `summary/missing-attacks.csv` for the attacks which are missing
    Note that all files are overwritten if they already exists
    """
    
    records = []
    missing_records = []
    field_map = {
    "WeightDecay": WeightDecayMap,
    "EpochNum": EpochNumsList,
    "Dataset": DataDict,
    "TrainSize": TrainSizeList,
    "Loss": LossDict,
    "Scheduler?": SchedulerList,
    "Arch": ArchToWidthDict
    }

    fields = list(field_map.keys())
    setups = product(*[field_map[field] for field in fields])
    for setup in setups:
        setup_dict = get_setup_dict(setup, fields)
        if setup_dict["Dataset"] not in LossDict[setup_dict["Loss"]] \
            or setup_dict["Arch"] not in DataDict[setup_dict["Dataset"]]:
            continue
        lr = 0.1
        if setup_dict["Arch"] == "alexnetwol" and setup_dict["Loss"] == "ce":
            lr = 0.01
        setup_dict["lr"] = lr
        width_to_setup_dict = {}
        for width in ArchToWidthDict[setup_dict["Arch"]]:
            curr_setup_dict = copy.deepcopy(setup_dict)
            width_to_setup_dict[width] = curr_setup_dict
            curr_setup_dict["Width"] = width
            model_name = "{dataset}-{arch}-w_{width}-ntrain_{tsize}-N_3".format(
            dataset=setup_dict["Dataset"],
            arch=setup_dict["Arch"],
            tsize=setup_dict["TrainSize"],
            width = curr_setup_dict["Width"])
            if setup_dict["lr"] == 0.01:
                model_name = "{model_name}-less_lr".format(model_name=model_name)
            curr_setup_dict["ModelName"] = model_name
            handle_model_stats(base_dir, curr_setup_dict)
            if (setup_dict["WeightDecay"] == "5e4"):
                handle_memguard_acc(base_dir, curr_setup_dict, missing_records)

        for attack_name in MlleakAttacksDict:
            # add the stats for mlleak-attack and mlleak_top3-attack
            fieldname_prefix = MlleakAttacksDict[attack_name]
            handle_mlleaks_attack(base_dir, attack_name, fieldname_prefix, width_to_setup_dict, setup_dict, missing_records)
        
        handle_oak17_attack(base_dir, width_to_setup_dict, missing_records)
        handle_threshold_attack(base_dir, width_to_setup_dict, missing_records)

        for width in width_to_setup_dict:
            # Upating the value of WeightDecay to actual number instead of abbreviation 5e3 or 5e4
            curr_setup_dict = width_to_setup_dict[width]
            curr_setup_dict["WeightDecay"] = WeightDecayMap[curr_setup_dict["WeightDecay"]]
            records.append(create_record_from_dict(curr_setup_dict, RECORD_FIELDS))
    
    full_summary_path = os.path.join(
        summary_dir,
        "full-summary.csv")
    missing_records_path = os.path.join(
        summary_dir,
        "missing-attacks.csv")

    write_records(full_summary_path, records, RECORD_FIELDS)
    write_records(missing_records_path, missing_records, MISSING_RECORD_HEADER)

def write_records(out_path, records, fieldnames):
    with open(out_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for record in records:
            writer.writerow(record)
    print("Done! ... %s" % out_path)

def create_summary_directories(summary_dir):
    if not(os.path.exists(summary_dir) and os.path.isdir(summary_dir)):
        os.mkdir(summary_dir)
    for attack_name in MlleakAttacksDict:
        attack_dir = os.path.join(summary_dir, attack_name)
        if not(os.path.exists(attack_dir) and os.path.isdir(attack_dir)):
            os.mkdir(attack_dir)


if __name__ == '__main__':
    # base_dir is "/mnt/teogang/bigger"
    GPU_ID = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    base_dir = sys.argv[1]
    summary_dir = sys.argv[2]
    create_summary_directories(summary_dir)
    loss_file = os.path.join(summary_dir, "detailed_losses.pkl")
    loss_dict = utils.read_pkl(loss_file) if os.path.exists(loss_file) else {}
    memguard_loss_file = os.path.join(summary_dir, "memguard_detailed_losses.pkl")
    memguard_loss_dict = utils.read_pkl(memguard_loss_file) if os.path.exists(memguard_loss_file) else {}
    parse_all_attack_acc(base_dir, summary_dir)
