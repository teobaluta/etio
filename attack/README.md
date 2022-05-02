## Membership Inference Attacks

**1. MLLeaks Attack**

### How to run?

First, install `ml_privacy_meter` by running the `pip install -e .` in the root directory of
the project after you've installed the dependencies.
```bash
python shadow_model_attack.py 
--data_tag cifar10
--torch_target_model_path root_dir/models/models-wd_5e3/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/models/final_models/final_model_76.pkl 
--torch_shadow_model_path root_dir/models/models-wd_5e3/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/models/final_models/final_model_55.pkl 
--config_file root_dir/models/models-wd_5e3/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/config.ini
--idx_file root_dir/datasets/prepare_data-data_cifar10-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz
--data_dir root_dir/datasets/
```
### Using the scripts

1. Pick random tuples of (target, shadow, nm) ids. The target model id is the model we are
	 attacking. Its corresponding training dataset is the member dataset used for evaluation. The
	 shadow model id is the model we use as a shadow model. Its corresponding training dataset is the
	 member dataset used for training the training the attack model. The non-member (nm) model id is
	 a different model whose training dataset we are going to use as non-members to train the attack
	 model (with the shadow dataset). We use this because we know it is disjoint from the target and
	 shadow datasets.
	 To generate the file, run the `gen_conf.py`. Run `--help` to check the options.
	 For example:
	 `python gen_conf.py 10 1 3 --outfile shadow_tuples-1k.conf`
2. Create a file containing the architecture options for a model. For example, if  `train_size width`
You can pass this file to the script and it will run `ml-leaks` and `ml-leaks-label` for all of
these architectures.
3. Use the `commands_to_run.sh`. **Make sure that the paths set in the script point to where the models and dataset splits are.**

Example of running the `commands_to_run.sh`
```
./commands_to_run.sh shadow_tuples-1k.conf arch_opt.conf cifar10 resnet34 w_scheduler mse logs/
```
For example, `arch_opt.conf` contains:
```
1 8
1 4
```

This means the script will run the attacks for Resnet34 trained with 1k samples for both width 8 and 4.

**2. Shadow Model Attack**

Reference: https://github.com/csong27/membership-inference/blob/a056ddc9599c1f7abd253722622f6b533b7c1925/attack.py#L85

### How to run?

```
python3 shokri_shadow_model_attack.py 
--model_dir root_dir/models/models-wd_5e3/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/models
--target_model_id 1 
--shadow_model_ids 2~50
--config_file root_dir/models/models-wd_5e3/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/models/config.ini
--idx_file root_dir/datasets/prepare_data-data_cifar10-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz --data_dir root_dir/DATASETS/ 
--attack_config_file root_dir/oak17-attacks/oak17-attacks-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/ 
--gpu_id 0 
--seeds 0
```

**3. Threshold Attack**
Reference: Adversary 1 from https://arxiv.org/pdf/1709.01604.pdf

### How to run?

```bash
python3 threshold-csf.py <dataset-path> <setup_dir_path> <gpu_id>
```

Example:
```bash
python3 threshold-csf.py root_dir/datasets/ root_dir/models/models-wd_5e3/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3  0
```

Executing this script will save the statistics for the threshold attack in a file `results-threshold_csf.pkl` inside the base directory of the model.

