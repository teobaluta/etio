## Membership Inference Attacks

**1. MLLeaks Attack**
### How to run?

First, install `ml_privacy_meter` by running the `pip install -e .` in the root directory of
the project after you've installed the dependencies.
```bash
python shadow_model_attack.py --torch_shadow_model_path ~/castle-data/cifar10-resnet34-w_8-ntrain_1k-N_3/final_model_55.pkl --torch_target_model_path ~/castle-data/cifar10-resnet34-w_8-ntrain_1k-N_3/final_model_1.pkl --config_file ../sample_config_file.conf --idx_file ~/castle-data/prepare_data-data_cifar10-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz --data_dir ~/DATASETS/ --data_tag cifar10
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

### How to run?

```
python3 shokri_shadow_model_attack.py --model_dir ~/data/cifar10-resnet34-w_8-ntrain_1k-N_3/models
--target_model_id 1 --shadow_model_id 2~50 --seeds 0 
--config_file ../sample_config_file.conf --idx_file ~/data/datasets/prepare_data-data_cifar10-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz --data_dir ~/DATASETS/cifar10
```

