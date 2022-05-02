## Membership Inference Attacks

**1. MLLeaks Attack**

First, install `ml_privacy_meter` by running the `pip install -e .` in the root directory of
the project after you've installed the dependencies.

### How to run for a single target model?

To run the attack for a single model with a given target model and shodow model, use the following command:
```bash
root_dir="path/to/main/data/folder"
python shadow_model_attack.py --data_tag cifar10 --torch_target_model_path $root_dir/models/models-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/models/final_models/final_model_76.pkl --torch_shadow_model_path $root_dir/models/models-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/models/final_models/final_model_55.pkl  --config_file $root_dir/models/models-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/config.ini --data_tag cifar10 --idx_file $root_dir/datasets/prepare_data-data_cifar10-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz --data_dir $root_dir/datasets/ --attack_model_output_dir $root_dir/mlleaks-attacks/mlleaks-attacks-wd_5e4/epoch_500/attack_models
```

The script will output the logs to the standart output. To save the logs, redirect the output to a log file inside a folder `<mlleaks-attack-model_dir>`. See the script `commands_to_run.sh` to check how we store the logs.

Example, `$root_dir/mlleaks-attacks/mlleaks-attacks-wd_5e4/epoch_500/logs/cifar10-resnet34-1k-w_scheduler-ce/cifar10-resnet34-w_8-1k-target_96-shadow_93-nm_91-ml_leaks_label-w_scheduler-ce.log`.

### How to run for multiple (target,shadow) model pairs?

Pick random tuples of (target, shadow, nm) ids. The target model id is the model we are
	 attacking. Its corresponding training dataset is the member dataset used for evaluation. The
	 shadow model id is the model we use as a shadow model. Its corresponding training dataset is the
	 member dataset used for training the training the attack model. The non-member (nm) model id is
	 a different model whose training dataset we are going to use as non-members to train the attack
	 model (with the shadow dataset). We use this because we know it is disjoint from the target and
	 shadow datasets.
	 To generate the file, run the `gen_conf.py`. Run `--help` to check the options.
	 For example:
	 `python gen_conf.py 10 1 3 --outfile shadow_tuples-1k.conf`

For example, `arch_option/shadow_tuples-1k.conf` contains:
```
32 26 20
7 40 17
47 5 44
...
```

#### Using Script

Using a shell script, we can then run the mlleaks attack for each tuple in the file.

1. Create a file containing the architecture options for a model. For example, if  `train_size width`
You can pass this file to the script and it will run `ml-leaks` and `ml-leaks-label` for all of
these architectures.

2. You can use/check the sample script `commands_to_run.sh`  to understand directory structure of the logs for the attack.
**Make sure that all the environment variables are set correctly before running the script**

Example of running the `commands_to_run.sh`
```
./commands_to_run.sh shadow_tuples-1k.conf arch_opt.conf cifar10 resnet34 w_scheduler mse logs/
```
For example, `arch_option/arch_options-1k.conf` contains:
```
1 8
1 4
```

### How to parse the logs?

For our parsers to work properly, the logs for all widths of a setup/architecture should be stored in a single folder. We assume that log files are named appropriately, as required by our parser. `analyze_attacks/parse_mlleaks.py`.

For example, for the model `$root_dir/models/models-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/`, we store the logs in the folder `$root_dir/mlleaks-attacks/mlleaks-attacks-wd_5e4/epoch_500/logs/cifar10-resnet34-1k-w_scheduler-ce/`, which is also shared by the other `models/models-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_*-ntrain_1k-N_3/` for all valid widths. Our parsers automatically create the attack results/summary for each architecture-width.

An example of a log file name that our parser assumes is `cifar10-resnet34-w_8-1k-target_96-shadow_93-nm_91-ml_leaks_label-w_scheduler-ce.log`.


**2. Oakland 2017 Shadow Model Attack**

See Reference: https://github.com/csong27/membership-inference/blob/a056ddc9599c1f7abd253722622f6b533b7c1925/attack.py#L85

### How to run?

```bash
root_dir="path/to/main/data/folder"
python3 oakland_shadow_model_attack.py  --model_dir $root_dir/models/models-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/models --target_model_id 1 --shadow_model_ids 2~50 --config_file $root_dir/models/models-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3/config.ini --idx_file $root_dir/datasets/prepare_data-data_cifar10-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz --data_dir $root_dir/datasets/ --attack_config_file $root_dir/oak17-attacks/oak17-attacks-wd_5e4/epoch_500/with_scheduler-ce/shadow_attack_cifar10-resnet34-w_8-ntrain_1k-N_3-target_1-shadow_2~50-seeds_0\,1/attack_model_config.ini --gpu_id 0 --seeds 0 --data_tag cifar10
```

**3. Threshold Attack**

See Reference: Adversary 1 from https://arxiv.org/pdf/1709.01604.pdf

### How to run?

```bash
python3 threshold-csf.py <dataset-path> <setup_dir_path> <gpu_id>
```

Example:
```bash
python3 threshold-csf.py root_dir/datasets/ root_dir/models/models-wd_5e4/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3  0
```

Executing this script will save the statistics for the threshold attack in a file `results-threshold_csf.pkl` inside the base directory of the model.

