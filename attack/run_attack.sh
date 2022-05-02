#!/bin/bash
DATASETS=$root_dir/datasets
# Added environment variables
if [[ "$#" -lt 7 ]]; then
	echo "Expecting arguments dataset, model name, train set size, width, target model_id, shadow model id,
	nm model id"
	echo "E.g., ./run_attack.sh cifar10 resnet34-w_8 1 8 1 3 2"
	exit
fi

max_output_shape=$max_output_shape
echo "Using max_output_shape = $max_output_shape"
force_softmax_for_ce=1
dataset=$1
model=$2
train_size=$3
width=$4
TARGET_MODEL_ID=$5
model_id=$6
NM_MODEL_ID=$7
attack_type="privacy-meter"


if [[ "$#" == 8 ]]; then
	if [[ $8 == "privacy_meter" || $8 == "ml_leaks" || $8 == "ml_leaks_label" ]]; then
		attack_type=$8
	fi
elif [[ "$#" -gt 8 ]]; then
	attack_type=$9
fi

if [[ $train_size == 1 ]]; then
	dataset_path=$DATA_DIR/prepare_data-data_$dataset-ntrial_3-train_size_1000-test_size_10000-strategy_disjoint_split-seed_0.npz
	eval_nm_size=1000
	ntrain=1k
elif [[ $train_size == 5 ]]; then
	dataset_path=$DATA_DIR/prepare_data-data_$dataset-ntrial_3-train_size_5000-test_size_10000-strategy_disjoint_split-seed_0.npz
	eval_nm_size=5000
	ntrain=5k
fi

if [[ "$#" == 8 && $8 == 'target' ]]; then
	python shadow_model_attack.py --torch_shadow_model_path $model_path/final_model_$model_id.pkl \
	--torch_target_model_path $model_path/final_model_$TARGET_MODEL_ID.pkl \
	--config_file $config_file --idx_file $dataset_path \
	--data_dir $DATASETS --data_tag $dataset --eval_nm_size $eval_nm_size \
	--ntrain $ntrain --shadow_model_id $model_id --target_attack \
	--target_model_id $TARGET_MODEL_ID  --nm_model_id $NM_MODEL_ID \
	--attack_type $attack_type \
	--loss_fn $loss_fn \
	--attack_model_output_dir $attack_output_dir \ 
	--max_output_shape $max_output_shape \
	--force_softmax_for_ce $force_softmax_for_ce
else
	python shadow_model_attack.py --torch_shadow_model_path $model_path/final_model_$model_id.pkl \
	--torch_target_model_path $model_path/final_model_$TARGET_MODEL_ID.pkl \
	--shadow_model_id $model_id \
	--config_file $config_file --idx_file $dataset_path \
	--data_dir $DATASETS --data_tag $dataset --eval_nm_size $eval_nm_size \
	--ntrain $ntrain \
	--target_model_id $TARGET_MODEL_ID  --nm_model_id $NM_MODEL_ID \
	--attack_type $attack_type \
	--loss_fn $loss_fn \
	--attack_model_output_dir $attack_output_dir \
	--max_output_shape $max_output_shape \
	--force_softmax_for_ce $force_softmax_for_ce
fi