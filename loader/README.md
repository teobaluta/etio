# Estimator

## Prepare Training Subsets

```bash
python prepare_dataset.py   \
    -d <data_tag>   \
    -o <out_dir>    \
    -trds <size_of_training_subset>   \
    -teds <size_of_testing_set>   \
    -t <trial_num>  \
    -rs <random_seed>   \
    -s <strategy>
```

- `data_tag`: Dataset name. Currently we only support `mnist`, `fmnist` and `cifar10`.
- `out_dir`: Path to the output directory.
- `size_of_training_subset`: Number of samples in each training set. (Default: `1000`)
- `size_of_testing_set`: Number of samples used for computing E_(x,y) (Default: `100`))
- `trial_num`: Number of trials. It can be number of models to train if using `random` strategies (e.g., `random-collision` and `random-no_collision`) or number of random splits if using `disjoint_split` strategy. (Default: `10`)
- `random_seed`: Seed for randomization. (Default: `0`)
- `strategy`: How to construct training sets for training models. Currently we only support `random-collision`, `random-no_collision` and `disjoint_split` (Default: `disjoint_split`) 

Instead of creating multiple subsets for training, for saving the space, we save the indexes of training samples being selected for each training subset in the `out_dir`. The file is named with following format: `train_idx_sets-data_<data_tag>-ntrial_<trial_num>-train_size_<size_of_training_subset>-test_size_<size_of_testing_set>-strategy_<strategy>-seed_<random_seed>.npy`
