#!/bin/bash


mkdir -p data/test_results
mkdir -p data/test_predictions

RESULT_NAME=train_FYNet_1_freqs_noiseless

DATA_BASE_DIR=/net/projects/willettlab/meliao/recursive-linearization/dataset
RESULTS_FP=data/results/$RESULT_NAME.txt
MODELS_DIR=/net/projects/willettlab/meliao/recursive-linearization/models/$RESULT_NAME
OUTPUT_SUMMARY_FP=data/test_results/$RESULT_NAME.yaml
OUTPUT_PREDICTIONS_DIR=data/test_predictions/$RESULT_NAME


echo "DATA_BASE_DIR: $DATA_BASE_DIR"
echo "RESULTS_FP: $RESULTS_FP"
echo "MODELS_DIR: $MODELS_DIR"

python train_MFISNet_Fused.py \
    --data_dir_base $DATA_BASE_DIR \
    --data_input_nus 16 \
    --model_weights_dir $MODELS_DIR \
    --train_results_fp $RESULTS_FP \
    --truncate_num 10000 \
    --truncate_num_val 1000 \
    --seed 35675 \
    --n_cnn_1d 3 \
    --n_cnn_2d 3 \
    --n_cnn_channels_1d 24 \
    --n_cnn_channels_2d 24 \
    --kernel_size_1d 60 \
    --kernel_size_2d 7 \
    --merge_middle_freq_channels true \
    --polar_padding true \
    --noise_to_signal_ratio 0 \
    --n_epochs 100 \
    --n_epochs_per_log 5 \
    --lr_init 1e-3 \
    --eta_min 1e-3 \
    --weight_decay 1e-3 \
    --batch_size 16 \
    --big_init \
    --debug \
    --wandb_mode disabled


python eval_MFISNet_Fused.py \
    --data_dir_base $DATA_BASE_DIR \
    --data_input_nus 16 \
    --eval_on_test_set \
    --model_dir $MODELS_DIR \
    --training_results_fp $RESULTS_FP \
    --training_results_key eval_rel_l2 \
    --test_output_summary_fp $OUTPUT_SUMMARY_FP \
    --test_output_predictions_dir $OUTPUT_PREDICTIONS_DIR