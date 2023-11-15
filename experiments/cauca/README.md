# Experiments for Causal Component Analysis (CauCA)

## Overview

The main entry point for the experiments is the `experiments/cauca/main.py` script.
It can be used to run the experiments for the different datasets and methods.
To run the experiments, use
```bash
python -m experiments.cauca.main [-h] [--max-epochs MAX_EPOCHS] [--accelerator ACCELERATOR] [--batch-size BATCH_SIZE] [--lr LR]
               [--checkpoint-root-dir CHECKPOINT_ROOT_DIR] [--noise-shift-type {mean,std}]
               [--check-val-every-n-epoch CHECK_VAL_EVERY_N_EPOCH] [--dgp DGP] [--k-flows K_FLOWS]
               [--k-flows-cbn K_FLOWS_CBN] [--model {linear,nonlinear,naive}] [--seed SEED]
               [--training-seed TRAINING_SEED] [--mixing {linear,nonlinear}] [--scm {linear,location-scale}]
               [--n-nonlinearities N_NONLINEARITIES] [--learn-scm-params | --no-learn-scm-params]
               [--lr-scheduler {None,cosine}] [--lr-min LR_MIN] [--scm-coeffs-low SCM_COEFFS_LOW]
               [--scm-coeffs-high SCM_COEFFS_HIGH] [--scm-coeffs-min-abs-value SCM_COEFFS_MIN_ABS_VALUE] [--snr SNR]
               [--adjacency-misspec | --no-adjacency-misspec] [--function-misspec | --no-function-misspec]
               [--net-hidden-layers NET_HIDDEN_LAYERS] [--net-hidden-layers-cbn NET_HIDDEN_LAYERS_CBN]
               [--net-hidden-dim NET_HIDDEN_DIM] [--net-hidden-dim-cbn NET_HIDDEN_DIM_CBN]
               [--fix-mechanisms | --no-fix-mechanisms]
               [--fix-all-intervention-targets | --no-fix-all-intervention-targets]
               [--nonparametric-base-distr | --no-nonparametric-base-distr] [--wandb | --no-wandb]
               [--wandb-project WANDB_PROJECT]
```

## Plotting the results

We suggest using [weights and biases (wandb)](https://wandb.ai/) to track the experiments and save the results to csv files.
The experiment script can be run without wandb (using `--no-wandb`), but the results will not be saved to any output files.
A custom logger would need to be implemented in `experiments/cauca/main.py` to save the results to csv files.

The csv files generated for the results shown in the paper can be found in the `experiments/cauca/results/data` directory.
The results can be plotted using the `experiments/cauca/results/create_figures.ipynb` notebook.

In order to install the required packages, run
```bash
pip install -e ".[plots]"
```
from the root directory of the repository. The notebooks are tracked in version control using
[jupytext](https://github.com/mwouts/jupytext).
In order to set up the `.ipynb` files, run
```bash
jupytext experiments/cauca/results/create_figures.py --to .ipynb
```
Then execute the notebook.