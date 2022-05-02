# Estimator

## Get Model Statistics 

To compute the bias-square, variance, average train loss, etc., use the script `estimate.py`. 

### Speicfy the setup

You need the path to the folder that has prepared datasets, and the directory path to the model. 

```bash
python3 estimate.py <path-to-prepared-datasets> <setup_dir_path> <gpu_id>
```

Example:
```bash
python3 estimate.py root_dir/datasets/ root_dir/models/models-wd_5e3/epoch_500/with_scheduler-ce/cifar10-resnet34-w_8-ntrain_1k-N_3  0
```

Executing this scrip will create `stats.pkl` in the model directory that stores a dictionary with all the computed statistics for this model.

## Get Centroid

To compute the centroid, use the script `get_centroid.py` in two modes. All of the commands above are run from the `analyze` folder.


### Specify the setups

First, the `one_setup` mode where we can manually specify a list of settings (i.e., arch, width, dataset, etc.) for which to compute the centroid.

For example:
```
python get_centroid.py one_setup --data_dir root_dir/datasets/ --model_dirs root_dir/models/models-wd_5e3/  --data_idx root_dir/datasets/prepare_data-data_cifar10-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz --losses ce --datasets cifar10 --archs resnet34 --widths 256 --ntrains 1k --repeat_num 3 --epoch_nums 500 --schedulers with_scheduler --model_num_per_repeat 50
```
To run for the models trained without scheduler, simply change the option `--schedulers with_scheduler` to `--schedulers with_scheduler,without_scheduler`.

Second, to run for all of our training setups (which we have specified in the `get_centroid.py`) given the path to the models and the path to the data directory (where the datasets and the index files exist) use the `all` mode.

For example:
```
 python get_centroid.py all --data_dir root_dir/datasets/ --model_dirs root_dir/models/models-wd_5e3/
```

These scripts first save the logits of the last layer (? check if logits or softmax) in a folder `last_layer` created in each model's directory. Then, it computes the centroid distance and appends it to a file `centroid_distance.txt`. Which is why if you run this command multiple times x, you will find x rows.
The saved metrics correspond to the centroid distance computed over the whole prediction vector, over the top 10 elements in the prediction vector (? check this, or if 10 is the total number, all of the prediction vector) or over the top 3 elements in the prediction vector.

### Collect the results

To collect the results in the format that we use throughout the project, use the `python gather_centroid.py <model_dir> <out_dir>`.
The `centroid_distance.txt` from every setup in `<model_dir>` is aggregated in a single file
`<out_dir>/stats-centroid_distance.txt`.

For example, corresponding to the command above for the `all` setup, we can gather all of the results using
```
python gather_centroid_distance.py root_dir/models/models-wd_5e3/
```


