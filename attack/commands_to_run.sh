#!/bin/bash
epoch_num=$epoch_num  # define epoch_num to be 500 or 400 as environment variable
max_output_shape=$max_output_shape # define it to be 3 or 10 as env. var
DATA_DIR="$root_dir/datasets"
MODEL_DIR="$root_dir/models/models-wd_5e4/epoch_500/"
CUDA_DEV=0
MAX_CUDA_DEV=2
CURR_PROCESS=0
MAX_PROCESS=4

if [[ "$#" -lt 8 ]]; then
	echo "Expecting ./commands_to_run.sh <shadow_split_tuples.conf> <arch_opt.conf> <dataset> <model_name>" \
	"w_scheduler/wo_scheduler mse/ce <log_dir> <attack_output_dir>"
	exit
fi

tuple_file=$1
arch_opt_file=$2
#e.g., cifar10
DATASET=$3
#e.g., densenet161
MODEL_NAME=$4
# if w_scheduler
w_scheduler=$5
# type of loss function, either mse or ce
loss_fn=$6
LOGS=$7
ATTACK_OUTPUT_PREFIX=$8

# File containing lines with the arch options as
# train_size width_size

training_width_args=()
while IFS='' read -r line; do
	training_width_args+=("$line")
done<$arch_opt_file

# File containing lines with the tuples that represent
# target model id, shadow model id, nonmember model id
target_shadow_args=()
while IFS='' read -r line; do
	target_shadow_args+=("$line")
done<$tuple_file

declare -a extra_args=(
	"ml_leaks"
	"ml_leaks_label"
	)

LOG_PREFIX=$LOGS"/"$DATASET"-"$MODEL_NAME"-"

for ts_elem in "${target_shadow_args[@]}"; do
	read -a strarr <<< "$ts_elem"
	for train_elem in "${training_width_args[@]}"; do
		read -a trainarr <<< "$train_elem"
		for extra in ${extra_args[@]}; do
			trains=${trainarr[0]}
			width=${trainarr[1]}
			target=${strarr[0]}
			shadow=${strarr[1]}
			nm=${strarr[2]}

			if [[ $w_scheduler == 'w_scheduler' ]]; then
				if [[ $loss_fn == 'mse' ]]; then
					MODELS_DIR=$model_base_dir/with_scheduler-mse
				else
					MODELS_DIR=$model_base_dir/with_scheduler-ce
				fi
				
				model_path=$MODELS_DIR/$DATASET-$MODEL_NAME-w_$width-ntrain_"$trains"k-N_3/models
				config_file=$MODELS_DIR/$DATASET-$MODEL_NAME-w_$width-ntrain_"$trains"k-N_3/config.ini
				if [[ $loss_fn == 'ce' ]]; then
					if [[ $MODEL_NAME == "alexnetwol" ]]; then
						model_path=$MODELS_DIR/$DATASET-$MODEL_NAME-w_$width-ntrain_"$trains"k-N_3-less_lr/models
						config_file=$MODELS_DIR/$DATASET-$MODEL_NAME-w_$width-ntrain_"$trains"k-N_3-less_lr/config.ini
					fi
				fi
			else
				if [[ $loss_fn == 'mse' ]]; then
					MODELS_DIR=$model_base_dir/without_scheduler-mse
				else
					MODELS_DIR=$model_base_dir/without_scheduler-ce
				fi
				model_path=$MODELS_DIR/$DATASET-$MODEL_NAME-w_$width-ntrain_"$trains"k-N_3/models
				config_file=$MODELS_DIR/$DATASET-$MODEL_NAME-w_$width-ntrain_"$trains"k-N_3/config.ini
				if [[ $loss_fn == 'ce' ]]; then
					if [[ $MODEL_NAME == "alexnetwol" ]]; then
						model_path=$MODELS_DIR/$DATASET-$MODEL_NAME-w_$width-ntrain_"$trains"k-N_3-less_lr/models
						config_file=$MODELS_DIR/$DATASET-$MODEL_NAME-w_$width-ntrain_"$trains"k-N_3-less_lr/config.ini
					fi
				fi
			fi

			attack_output_dir="$ATTACK_OUTPUT_PREFIX/$DATASET-$MODEL_NAME-$w_scheduler-$loss_fn"
			cmd="config_file=$config_file model_path=$model_path DATA_DIR=$DATA_DIR attack_output_dir=$attack_output_dir loss_fn=$loss_fn max_output_shape=$max_output_shape  CUDA_VISIBLE_DEVICES=$CUDA_DEV ./run_attack.sh $DATASET $MODEL_NAME"-w_"$width ${train_elem} ${ts_elem} --attack_type $extra > $LOG_PREFIX"w_"$width"-"$trains"k-target_"$target"-shadow_"$shadow"-nm_"$nm"-"$extra"-$w_scheduler-$loss_fn.log" 2>&1 "
			CUDA_DEV=$((CUDA_DEV+1))
			CURR_PROCESS=$((CURR_PROCESS+1))
			cmd=$(echo $cmd " & ")
			echo $cmd
			eval $cmd
			if [[ $CUDA_DEV == $MAX_CUDA_DEV ]]; then
				# We loop back when we exhaust the available GPUs
				CUDA_DEV=0
			fi
			if [[ $CURR_PROCESS == $MAX_PROCESS ]]; then
				# We will wait when we are running maximum allowed processes
				CURR_PROCESS=0
				echo "Running $MAX_PROCESS in background. Waiting for them to finish..."
				wait
			fi
		done
	done
done
