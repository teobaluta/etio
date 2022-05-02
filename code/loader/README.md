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


## Get Centroid

To compute the centroid, use the script `get_centroid.py` in two modes. All of the commands above are run from the `analyze` folder.


### Specify the setups

First, the `one_setup` mode where we can manually specify a list of settings (i.e., arch, width, dataset, etc.) for which to compute the centroid.

For example:
```
python get_centroid.py one_setup --data_dir /mnt/archive0/hitarth/datasets/ --model_dirs /mnt/archive0/hitarth/models-high_weight_decay/ --data_idx /mnt/archive0/hitarth/datasets/prepare_data-data_cifar100-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz --losses ce --datasets cifar100 --archs alexnetwol --widths 256 --ntrains 1k --repeat_num 3 --epoch_nums 500 --schedulers with_scheduler --model_num_per_repeat 50
```
To run for the models trained without scheduler, simply change the option `--schedulers with_scheduler` to `--schedulers with_scheduler,without_scheduler`.

Second, to run for all of our training setups (which we have specified in the `get_centroid.py`) given the path to the models and the path to the data directory (where the datasets and the index files exist) use the `all` mode.

For example:
```
 python get_centroid.py all --data_dir /mnt/archive0/hitarth/datasets/ --model_dirs /mnt/archive0/hitarth/models-high_weight_decay/
```

These scripts first save the logits of the last layer (? check if logits or softmax) in a folder `last_layer` created in each model's directory. Then, it computes the centroid distance and appends it to a file `centroid_distance.txt`. Which is why if you run this command multiple times x, you will find x rows.
The saved metrics correspond to the centroid distance computed over the whole prediction vector, over the top 10 elements in the prediction vector (? check this, or if 10 is the total number, all of the prediction vector) or over the top 3 elements in the prediction vector.

### Collect the results

To collect the results in the format that we use throughout the project, use the `python gather_centroid.py <model_dir> <out_dir>`.
The `centroid_distance.txt` from every setup in `<model_dir>` is aggregated in a single file
`<out_dir>/stats-centroid_distance.txt`.

For example, corresponding to the command above for the `all` setup, we can gather all of the results using
```
python gather_centroid_distance.py /mnt/archive0/hitarth/models-high_weight_decay/
```


