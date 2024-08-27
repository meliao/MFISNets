#!/bin/bash

which python

mkdir -p data/dataset/

python generate_scattering_files.py \
--n_samples 50 \
--n_pixels 192 \
--contrast 2.0 \
--output_fp data/dataset/scattering_objs/scattering_objs_0.h5 \
--background_max_freq 0.4 \
--background_max_freq 4.0 \
--spatial_domain_max 0.5 \
--gaussian_lpf_param 16.0 \
--seed 0 \
--dont_use_wandb
