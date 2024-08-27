#!/bin/bash

which python

python generate_measurement_files.py \
    --input_fp data/dataset/scattering_objs/scattering_objs_0.h5 \
    --output_fp data/dataset/measurements_nu_8/measurements_0.h5 \
    --nu_source_freq 8 \
    --batch_size 100 \
    --receiver_radius 100 \
    --start_idx 0 \
    --write_every_n 5 \
    --dont_use_wandb
