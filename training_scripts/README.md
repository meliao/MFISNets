This is a directory where training scripts for models with N_k = 3 and 5 live.

We provide scripts for the following models, using the best hyperparameters resulting from the hyperparameter grid search outlined in the paper:
- FYNet (baseline)
- MFISNet-Parallel
- MFISNet-Fused
- MFISNet-Refinement (main version)
- MFISNet-Refinement (without progressive refinement)
- MFISNet-Refinement (without homotopy)

The Wide-band Butterfly Network code is maintained by its authors, which can be found on GitHub at [https://github.com/borongzhang/ISP_baseline](https://github.com/borongzhang/ISP_baseline).

Each training script should be called from the root directory in a python environment specified by `env.yaml`, also found in the root directory.
