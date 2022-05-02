#!/bin/bash

DATA_DIR=/hdd/DATASETS
OUT_DIR=./ce-stats

if [[ $1 == "w_scheduler" ]]; then
	CONFIG_PATH=/hdd/castle-data/with_scheduler-ce
elif [[ $1 == "wo_scheduler" ]]; then
	CONFIG_PATH=/hdd/castle-data/
else
	echo "Not recognized either w_scheduler or wo_scheduler"
	exit
fi

declare -a models=(
	"cifar10-resnet34-w_2"
	"cifar10-resnet34-w_4"
	"cifar10-resnet34-w_8"
	"cifar10-resnet34-w_16"
	"cifar10-resnet34-w_32"
	"cifar10-densenet161-w_32"
	"cifar10-densenet161-w_16"
	"cifar10-densenet161-w_8"
	"cifar10-densenet161-w_4"
	"cifar10-densenet161-w_2"
)

for model in ${models[@]}; do
	python estimate_kurt.py --kurt --data_tag cifar10 --data_dir $DATA_DIR --idx_file \
		/hdd/castle-data/prepare_data-data_cifar10-ntrial_3-train_size_5000-test_size_10000-strategy_disjoint_split-seed_0.npz \
		--config_file $CONFIG_PATH/$model-ntrain_5k-N_3/config.ini > $OUT_DIR/$model-$1-kurtosis_test.log 2>&1 &

	python estimate_kurt.py --kurt --data_tag cifar10 --data_dir $DATA_DIR --idx_file \
		/hdd/castle-data/prepare_data-data_cifar10-ntrial_3-train_size_5000-test_size_10000-strategy_disjoint_split-seed_0.npz \
		--config_file $CONFIG_PATH/$model-ntrain_5k-N_3/config.ini --train > $OUT_DIR/$model-$1-kurtosis_train.log 2>&1
done
