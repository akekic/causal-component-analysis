# Experiments for Nonparametric Identifiability of Causal Representations from Unknown Interventions

[Link to paper](https://openreview.net/pdf?id=V87gZeSOL4).

## Overview

The main entry point for the experiments is the `experiments/nonparam_ident/main.py` script.
It can be used to run the experiments for the different datasets and methods.
To run the experiments, use
```bash
python -m experiments.nonparam_ident.main.py [-h] [--max-epochs MAX_EPOCHS] [--accelerator ACCELERATOR] [--batch-size BATCH_SIZE] [--lr LR] [--checkpoint-root-dir CHECKPOINT_ROOT_DIR]
               [--noise-shift-type {mean,std}] [--check-val-every-n-epoch CHECK_VAL_EVERY_N_EPOCH] [--dgp DGP] [--k-flows K_FLOWS] [--k-flows-cbn K_FLOWS_CBN]
               [--model {linear,nonlinear,naive}] [--seed SEED] [--training-seed TRAINING_SEED] [--mixing {linear,nonlinear}] [--scm {linear,location-scale}]
               [--n-nonlinearities N_NONLINEARITIES] [--learn-scm-params | --no-learn-scm-params] [--lr-scheduler {None,cosine}] [--lr-min LR_MIN]
               [--scm-coeffs-low SCM_COEFFS_LOW] [--scm-coeffs-high SCM_COEFFS_HIGH] [--scm-coeffs-min-abs-value SCM_COEFFS_MIN_ABS_VALUE] [--snr SNR]
               [--adjacency-misspec | --no-adjacency-misspec] [--intervention-target-misspec | --no-intervention-target-misspec]
               [--intervention-target-perm INTERVENTION_TARGET_PERM [INTERVENTION_TARGET_PERM ...]] [--net-hidden-layers NET_HIDDEN_LAYERS]
               [--net-hidden-layers-cbn NET_HIDDEN_LAYERS_CBN] [--net-hidden-dim NET_HIDDEN_DIM] [--net-hidden-dim-cbn NET_HIDDEN_DIM_CBN]
               [--fix-mechanisms | --no-fix-mechanisms] [--fix-all-intervention-targets | --no-fix-all-intervention-targets]
               [--nonparametric-base-distr | --no-nonparametric-base-distr] [--wandb | --no-wandb] [--wandb-project WANDB_PROJECT]
```

## Reproducing the results

Here we list the commands needed to produce the results shown in the paper.
<details>
<summary>Click to show full list of commands</summary>

- **Description:** parametric CauCA.
  - **Command:**
    ```bash
    python -m experiments.nonparam_ident.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 200 --k-flows 12 --dgp graph-2-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --scm-coeffs-low -10 --scm-coeffs-high 10 --scm-coeffs-min-abs-value 2 --no-fix-mechanisms
    ```
    where `S=0, ...,49` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/nonparam_ident/results/data/3_nonlin_cauca_2vars_10scm_min2.csv`
  - **Used in:** Figure 3.
- **Description:** parametric CauCA with intervention target and adjacency misspecification.
  - **Command:**
    ```bash
    python -m experiments.nonparam_ident.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 200 --k-flows 12 --dgp graph-2-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --scm-coeffs-low -10 --scm-coeffs-high 10 --scm-coeffs-min-abs-value 2 --adjacency-misspec --intervention-target-misspec --fix-all-intervention-targets
    ```
    where `S=0, ...,49` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/nonparam_ident/results/data/3_nonlin_cauca_both_misspec_2vars_10scm_min2.csv`
  - **Used in:** Figure 3.
- **Description:** parametric CauCA with intervention target misspecification.
  - **Command:**
    ```bash
    python -m experiments.nonparam_ident.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 200 --k-flows 12 --dgp graph-2-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --scm-coeffs-low -10 --scm-coeffs-high 10 --scm-coeffs-min-abs-value 2 --intervention-target-misspec --fix-all-intervention-targets
    ```
    where `S=0, ...,49` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/nonparam_ident/results/data/3_nonlin_cauca_int_misspec_2vars_10scm_min2.csv`
  - **Used in:** Figure 3.
- **Description:** parametric CauCA with adjacency misspecification.
  - **Command:**
    ```bash
    python -m experiments.nonparam_ident.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 200 --k-flows 12 --dgp graph-2-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --scm-coeffs-low -10 --scm-coeffs-high 10 --scm-coeffs-min-abs-value 2 --adjacency-misspec --fix-all-intervention-targets
    ```
    where `S=0, ...,49` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/nonparam_ident/results/data/3_nonlin_cauca_misspec_2vars_10scm_min2.csv`
  - **Used in:** Figure 3.
- **Description:** nonparametric CauCA.
  - **Command:**
    ```bash
    python -m experiments.nonparam_ident.main --seed <S> --training-seed <T> --model nonlinear --scm location-scale --nonparametric-base-distr --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-3-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --snr 10.0
    ```
    where `S=0, ...,19` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/nonparam_ident/results/data/3var_snr10_012.csv`
  - **Used in:** Figure 4.
- **Description:** nonparametric CauCA with intervention target permutation.
  - **Command:**
    ```bash
    python -m experiments.nonparam_ident.main --seed <S> --training-seed <T>  --intervention-target-perm <P> --model nonlinear --scm location-scale --nonparametric-base-distr --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-3-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --snr 10.0  --intervention-target-misspec
    ```
    where `S=0, ...,19` and `T=0, 1, 2` are used for the different seeds and `P ∈ {"0 2 1", "1 0 2", "1 2 0", "2 0 1", "2 1 0"}`
    are the different permutations of the intervention targets.
  - **Output file:** `experiments/nonparam_ident/results/data/3var_snr10_<P>.csv`
  - **Used in:** Figure 4.


</details>

## Plotting the results

We suggest using [weights and biases (wandb)](https://wandb.ai/) to track the experiments and save the results to csv files.
The experiment script can be run without wandb (using `--no-wandb`), but the results will not be saved to any output files.
A custom logger would need to be implemented in `experiments/nonparam_ident/main.py` to save the results to csv files.

The csv files generated for the results shown in the paper can be found in the `experiments/nonparam_ident/results/data` directory.
The results can be plotted using the `experiments/nonparam_ident/results/create_figures.ipynb` notebook.

In order to install the required packages, run
```bash
pip install -e ".[plots]"
```
from the root directory of the repository. The notebooks are tracked in version control using
[jupytext](https://github.com/mwouts/jupytext).
In order to set up the `.ipynb` files, run
```bash
jupytext experiments/nonparam_ident/results/create_figures.py --to .ipynb
```
Then execute the notebook.