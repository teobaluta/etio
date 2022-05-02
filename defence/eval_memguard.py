import numpy as np
import torch
import memguard
from tqdm import tqdm
import os
import pickle
import sys

def execute_attack_model(after_softmax, input_labels, attack_model, construct_input_fn):
    attack_model.eval()
    member_num = after_softmax.shape[0]
    preds = []
    with torch.no_grad():
        for member_id in tqdm(range(member_num), desc="Querying Attack Model"):
            sorted_input, sorted_idx_list = construct_input_fn(torch.tensor(after_softmax[member_id]).cuda(), input_labels[member_id])
            output = attack_model(sorted_input.reshape(1, -1))
            preds.append(torch.softmax(output, axis = 1)[0].cpu().numpy())
    preds = np.asarray(preds)
    return preds

def compute_inference_acc(epsilon_value, 
                          member_pred_vector, updated_member_pred_vector, member_after_softmax, updated_member_after_softmax,
                          nonmember_pred_vector, updated_nonmember_pred_vector, nonmember_after_softmax, updated_nonmember_after_softmax):
    member_inference_accuracy=0.0
    np.random.seed(100)  
    for i in tqdm(range(member_after_softmax.shape[0]), desc="Member"):
        distortion_noise=np.sum(np.abs(member_after_softmax[i,:]-updated_member_after_softmax[i,:]))
        p_value=0.0
        # print(member_pred_vector.shape)
        if np.abs(member_pred_vector[i][1]-0.5)<=np.abs(updated_member_pred_vector[i][1]-0.5):
            p_value=0.0
        else:
            p_value=min(epsilon_value/distortion_noise,1.0)

        if np.argmax(member_pred_vector[i]) == 1:
            member_inference_accuracy+=1.0-p_value
        if np.argmax(updated_member_pred_vector[i]) == 1:
            member_inference_accuracy+=p_value
    member_acc = member_inference_accuracy/(float(member_after_softmax.shape[0]))
    
    nonmember_inference_accuracy=0.0     
    for i in tqdm(range(nonmember_after_softmax.shape[0]), desc="Non-Member"):
        distortion_noise=np.sum(np.abs(nonmember_after_softmax[i,:]-updated_nonmember_after_softmax[i,:]))
        p_value=0.0
        if np.abs(nonmember_pred_vector[i][1]-0.5)<=np.abs(updated_nonmember_pred_vector[i][1]-0.5):
            p_value=0.0
        else:
            p_value=min(epsilon_value/distortion_noise,1.0)

        if np.argmax(nonmember_pred_vector[i]) == 0:
            nonmember_inference_accuracy+=1.0-p_value
        if np.argmax(updated_nonmember_pred_vector[i]) == 0:
            nonmember_inference_accuracy+=p_value
    nonmember_acc = nonmember_inference_accuracy/(float(nonmember_after_softmax.shape[0]))
    
    return member_acc, nonmember_acc

def eval_memguard(memguard_file, attack_model_path, arch, loss):
    print(memguard_file)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(memguard_file)), "memguard_acc")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    out_path = os.path.join(
        out_dir, os.path.basename(memguard_file).split(".")[0] + ".pkl"
    )
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            acc_dict = pickle.load(f)
        return acc_dict
    # memguard_file = "/mnt/archive0/shiqi/epoch_500/with_scheduler-mse/cifar10-densenet161-w_2-ntrain_1k-N_3/memguard/cifar10-densenet161-w_2-1k-target_model_id_14-shadow_id_33-nm_model_id_9-ml_leaks-shadow-loss_mse-attack_model.npz"
    # attack_model_path = "/home/hitarth/ml_leak_attacks/epoch_500-with_top3/attack_models/cifar10-densenet161-w_scheduler-mse/cifar10-densenet161-w_2-1k-target_model_id_14-shadow_id_33-nm_model_id_9-ml_leaks-shadow-loss_mse-attack_model.pth"
    # arch = "densenet161"
    # loss = "mse"
    # attack_type = "ml_leaks"
    attack_type = os.path.basename(memguard_file).split("-")[7]
    epsilon_value_list = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
    
    attack_model = torch.load(attack_model_path)
    data = np.load(memguard_file)
    
    member_after_softmax = data["member_after_softmax"]
    nonmember_after_softmax = data["nonmember_after_softmax"]
    
    updated_member_after_softmax = data["updated_member_after_softmax"]
    updated_nonmember_after_softmax = data["updated_nonmember_after_softmax"]
    
    member_labels = data["member_labels"]
    nonmember_labels = data["nonmember_labels"]
    
    construct_input_fn = memguard.ConstructInputFNList[attack_type]
    
    member_pred_vector = execute_attack_model(member_after_softmax, member_labels, attack_model, construct_input_fn)
    nonmember_pred_vector = execute_attack_model(nonmember_after_softmax, nonmember_labels, attack_model, construct_input_fn)
    
    updated_member_pred_vector = execute_attack_model(updated_member_after_softmax, member_labels, attack_model, construct_input_fn)
    updated_nonmember_pred_vector = execute_attack_model(updated_nonmember_after_softmax, nonmember_labels, attack_model, construct_input_fn)

    acc_dict = {}    
    for epsilon_value in epsilon_value_list:
        member_acc, nonmember_acc = compute_inference_acc(epsilon_value, 
                            member_pred_vector, updated_member_pred_vector, member_after_softmax, updated_member_after_softmax,
                            nonmember_pred_vector, updated_nonmember_pred_vector, nonmember_after_softmax, updated_nonmember_after_softmax,)
        acc_dict[epsilon_value] = {'MemberAcc': member_acc, 'NonmemberAcc': nonmember_acc}
    with open(out_path, "wb") as f:
        pickle.dump(acc_dict, f)
    return acc_dict

def main(target_dir, attack_dir):
    epoch_nums = [500]
    schedulers = ["with_scheduler", "without_scheduler"]
    scheduler_map = {
        "with_scheduler": "w_scheduler",
        "without_scheduler": "wo_scheduler"
    }
    losses = ["ce", "mse"]
    for epoch_num in epoch_nums:
        for scheduler in schedulers:
            for loss in losses:
                tmp = f"{target_dir}/epoch_{epoch_num}/{scheduler}-{loss}"
                if not os.path.isdir(tmp):
                    continue
                filenames = os.listdir(tmp)
                for filename in filenames:
                    dataset = filename.split("-")[0]
                    arch = filename.split("-")[1]
                    tmp2 = f"{tmp}/{filename}/memguard"
                    if os.path.isdir(tmp2):
                        tmp3 = os.listdir(tmp2)
                        for tmp4 in tmp3:
                            memguard_file = os.path.join(tmp2, tmp4)
                            tmp5 = tmp4.split(".")[0] + ".pth"
                            attack_model_path = f"{attack_dir}/epoch_{epoch_num}-with_top3/attack_models/{dataset}-{arch}-{scheduler_map[scheduler]}-{loss}/{tmp5}"
                            eval_memguard(memguard_file, attack_model_path, arch, loss)
        
    
    

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
