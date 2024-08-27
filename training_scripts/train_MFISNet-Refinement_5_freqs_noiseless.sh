#!/bin/bash



DATA_BASE_DIR=/net/projects/willettlab/meliao/recursive-linearization/dataset
MEAS_DIR_TRAIN_FRMT=/net/projects/willettlab/meliao/recursive-linearization/dataset/train_measurements_nu_{}
SCOBJ_DIR_TRAIN=/net/projects/willettlab/meliao/recursive-linearization/dataset/train_scattering_objs
MEAS_DIR_VAL_FRMT=/net/projects/willettlab/meliao/recursive-linearization/dataset/val_measurements_nu_{}
SCOBJ_DIR_VAL=/net/projects/willettlab/meliao/recursive-linearization/dataset/val_scattering_objs


RESULT_NAME=train_MFISNet-Refinement_5_freqs_noiseless



RESULTS_FP=data/results/$RESULT_NAME.txt
MODELS_DIR=/net/projects/willettlab/meliao/recursive-linearization/models/$RESULT_NAME
OUTPUT_SUMMARY_FP=data/test_results/$RESULT_NAME.yaml
OUTPUT_PREDICTIONS_DIR=data/test_predictions/$RESULT_NAME

python train_MFISNet_Refinement.py \
-meas_dir_train_frmt $MEAS_DIR_TRAIN_FRMT \
-scobj_dir_train $SCOBJ_DIR_TRAIN \
-meas_dir_val_frmt $MEAS_DIR_VAL_FRMT \
-scobj_dir_val $SCOBJ_DIR_VAL \
-model_weights_dir $MODELS_DIR \
-results_fp $RESULTS_FP \
-n_epochs_pretrain 50 \
-n_epochs_finetune 50 \
-n_epochs_per_log 5 \
-lr_init 1e-03 \
-lr_decrease_factor 0.25 \
-eta_min 1e-03 \
-n_cnn_channels_1d 24 \
-n_cnn_channels_2d 24 \
-n_cnn_1d 3 \
-n_cnn_2d 3 \
-kernel_size_1d 40 \
-kernel_size_2d 5 \
-dont_use_wandb \
-weight_decay 1e-03 \
-seed 1001 \
-batch_size 16 \
-dont_write_outputs \
-wavenumbers 1 2 4 8 16 \
-truncate_num 2000 \
-truncate_num_val 200 \

python eval_MFISNet_Refinement.py \
    --data_dir_base $DATA_BASE_DIR \
    --wavenumbers 1 2 4 8 16 \
    --model_dir $MODELS_DIR \
    --training_results_fp $RESULTS_FP \
    --training_results_key val_final_rel_l2 \
    --test_output_summary_fp $OUTPUT_SUMMARY_FP \
    --test_output_predictions_dir $OUTPUT_PREDICTIONS_DIR