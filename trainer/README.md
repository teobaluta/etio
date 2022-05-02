# Training Models

```bash
cd trainer
python train.py -d <data_tag> -dd <data_dir_path> -i <data_idx_file> -c <config_file> -mi <model_ids> -g <gpu_id>
```

- `data_tag`: The dataset used for training the models. Options: [`cifar100`, `cifar10`, `mnist`]
- `data_dir_path`: The path to the data directory.
- `data_idx_file`: The path to the dataset splitting information used for training.
- `config_file`: The model configuration file used for training.
- `model_ids`: The list of models for training. Example, `1~3` means that the script will train the models seperately using the first, the second and the third subset. The dataset subsets are generated using `../estimator/prepare_dataset.py` 
- `gpu_id`: The ID of GPU for computing the stats.

Output: The trained models can be found at the folder of `config_file`.
