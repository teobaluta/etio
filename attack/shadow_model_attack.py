import torch
import numpy as np
import os
import sys
import os
import shutil
sys.path.append('../')
from trainer import train
from trainer import models
import estimator
import estimator.prepare_dataset as prep_data
import tensorflow as tf
from tensorflow.compat.v1.train import Saver
from tensorflow import keras
import argparse
import pickle as pkl

import ml_leaks
import privacy_meter

# Use only one GPU to easily avoid having tensors on different GPUs
TARGET_MODEL_ID = 1
NONMEMBER_MODEL_ID = 2
ATTACK_MODELS_DIR='../ml_leaks-attack_models'

def load_dataset(data_tag, idx_file, train_config, data_dir, model_id):
    idx_data = estimator.utils.read_npy(idx_file)
    train_set, test_set = train.load_complete_data(data_tag, data_dir)
    train_loader = train.prepare_test_set(data_tag, train_set,
                                          idx_data['train_idx_sets'][model_id - 1],
                                          int(train_config['train_batch_size']))

    test_loader = train.prepare_test_set(data_tag, test_set,
                                         idx_data['test_idx_set'],
                                         int(train_config['test_batch_size']))
    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='arguments for the attack')
    parser.add_argument('--torch_shadow_model_path', required=True, type=str,
                        help='Path to the Pytorch model')
    parser.add_argument('--torch_target_model_path', required=True, type=str,
                        help='Path to the Pytorch model')
    parser.add_argument('--data_tag', type=str, required=True,
                        choices=list(prep_data.RegisteredDatasets.keys()),
                        help='Dataset name')
    parser.add_argument('--idx_file', type=str, required=True,
                        help='path to to the data index file')
    parser.add_argument('--config_file', type=str, required=True, help='Model config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory path.')
    parser.add_argument('--target_model_id', type=int, default=TARGET_MODEL_ID,
                        help='Target model id')
    parser.add_argument('--shadow_model_id', type=int, default=3, help='Shadow model idx')
    parser.add_argument('--nm_model_id', type=int, default=NONMEMBER_MODEL_ID, help='Take this '
                        ' models training set as the shadow models non-member training set')
    parser.add_argument('--eval_nm_size', default=1000, type=int,
                        help='Number of non-members to put in the evaluation set')
    parser.add_argument('--ntrain', type=str, default='1k', help='Number of training samples for shadow'
                        ' model/target model. Used to create the folder name of datasets')
    parser.add_argument('--target_attack', action='store_true', default=False, help='If set to true, do an attack'
                        ' with the target model directly.')
    parser.add_argument('--attack_type', choices=['ml_leaks', 'ml_leaks_label', 'privacy_meter'],
                        default='attack-type', type=str, help='Type of attack')
    parser.add_argument('--loss_fn', choices=['mse', 'ce'], default='mse', type=str,
                        help='Type of loss function')
    parser.add_argument('--attack_model_output_dir', type=str, default=ATTACK_MODELS_DIR,
                        help='Save the attack model')
    parser.add_argument('--max_output_shape', type=int, default=10,
                        help="Use prediction vector's top (max_output_shape) if the output shape is greater than max_output_shape")
    parser.add_argument('--force_softmax_for_ce', type=bool, default=False,
                        help="Use softmax on the prediction vector if the loss is CE")
    args = parser.parse_args()
    print('ARGS')
    print(args)
    train_config, model_config = train.load_config(args.config_file)
    if 'width' not in model_config:
        model_width = 'mlleak'
    else:
        model_width = model_config['width']
    if args.target_attack:
        attack_model_name = args.data_tag + '-' + model_config['model_tag'] + \
            '-w_' + model_width  + '-' + args.ntrain + \
            '-target_model_id_' + str(args.target_model_id) + \
            '-shadow_id_' + str(args.shadow_model_id) + \
            '-nm_model_id_' + str(args.nm_model_id) + '-' + args.attack_type + '-target' + \
            '-loss_' + str(args.loss_fn)
    else:
        attack_model_name = args.data_tag + '-' + model_config['model_tag'] + \
            '-w_' + model_width  + '-' + args.ntrain + \
            '-target_model_id_' + str(args.target_model_id) + \
            '-shadow_id_' + str(args.shadow_model_id) + \
            '-nm_model_id_' + str(args.nm_model_id) + '-' + args.attack_type + '-shadow' + \
            '-loss_' + str(args.loss_fn)

    print('Attack name {}'.format(attack_model_name))

    if args.force_softmax_for_ce and train_config["loss"].lower() == "ce":
        print("Using softmax for the CE model")
        model_config["w_softmax"] = "yes" # for CE loss, add the softmax while loading the model because we are not training the model

    target_model = models.MODELS[model_config['model_tag']](model_config).cuda()
    target_model.load_state_dict(torch.load(args.torch_target_model_path))
    target_model.eval()
    print('Pytorch target model {} loaded'.format(args.torch_target_model_path))

    if not args.target_attack:
        shadow_model = models.MODELS[model_config['model_tag']](model_config).cuda()
        shadow_model.load_state_dict(torch.load(args.torch_shadow_model_path))
        shadow_model.eval()
        print('Pytorch shadow model {} loaded'.format(args.torch_shadow_model_path))
    else:
        shadow_model = target_model
        print('Performing attack with TARGET_MODEL')

    member_shadow_loader, _ = load_dataset(args.data_tag, args.idx_file, train_config,
                                           args.data_dir, args.shadow_model_id)
    non_member_shadow_loader, test_loader = load_dataset(args.data_tag, args.idx_file,
                                                         train_config, args.data_dir,
                                                         args.nm_model_id)
    # Check accuracies
    if args.loss_fn == 'mse':
        test_fn = train.test_mse
    else:
        test_fn = train.test_ce

    train_loss, train_acc = test_fn(member_shadow_loader, shadow_model,
                                    int(model_config["out_dim"]))
    print('Shadow model accuracy on member (training) set:{}'.format(train_acc))

    nm_loss, nm_acc = test_fn(non_member_shadow_loader, shadow_model, int(model_config["out_dim"]))
    test_loss, test_acc = test_fn(test_loader, shadow_model, int(model_config["out_dim"]))
    print('Shadow model accuracy on non-member set:{}'.format(nm_acc))
    print('Shadow model accuracy on test (10k) set:{}'.format(test_acc))

    # non_member_target_loader is basically the testing set (max 10k)
    member_target_loader, non_member_target_loader = load_dataset(args.data_tag, args.idx_file,
                                                                  train_config,
                                                                  args.data_dir, args.target_model_id)
    # Check accuracies
    train_loss, train_acc = test_fn(member_target_loader, target_model, int(model_config["out_dim"]))
    test_loss, test_acc = test_fn(non_member_target_loader, target_model, int(model_config["out_dim"]))
    print('Target model train acc:{}'.format(train_acc))
    print('Target model test acc:{}'.format(test_acc))

    test_loss, test_acc = test_fn(member_target_loader, shadow_model, int(model_config["out_dim"]))
    print('Shadow model accuracy on eval members:{}'.format(test_acc))

    print('len(members_shadow)={}'.format(len(member_shadow_loader.dataset)))
    print('len(non_members_shadow)={}'.format(len(non_member_shadow_loader.dataset)))
    print('len(members_target)={}'.format(len(member_target_loader.dataset)))
    print('len(non_members_target)={}'.format(len(non_member_target_loader.dataset)))

    w_label = False
    if args.attack_type == 'privacy-meter':
        # the shadow model loader itself needs to be modified

        privacy_meter.blackbox_attack(target_model, shadow_model, member_shadow_loader,
                                      non_member_shadow_loader, member_target_loader,
                                      non_member_target_loader, attack_model_name,
                                      eval_nm_size=args.eval_nm_size)
        return

    if args.attack_type == 'ml_leaks_label':
        w_label = True
    else:
        w_label = False

    if not os.path.exists(args.attack_model_output_dir):
        os.mkdir(args.attack_model_output_dir)
    attack_model_name = os.path.join(args.attack_model_output_dir, attack_model_name + \
                                     '-attack_model.pth')
    ml_leaks.attack(target_model, shadow_model,
                    member_shadow_loader, non_member_shadow_loader,
                    member_target_loader, non_member_target_loader,
                    w_label, args.eval_nm_size, attack_model_name, args.max_output_shape)


if __name__ == "__main__":
    main()