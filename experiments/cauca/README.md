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

## Reproducing the results

Here we list the commands needed to produce the results shown in the paper.
<details>
<summary>Click to show full list of commands</summary>

- **Description:** parametric CauCA with two nonlinearities in the true mixing function.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 2 --lr-scheduler cosine --lr-min 1e-7
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/2_nonlin_cauca.csv`
  - **Used in:** Figure 4 (c).
- **Description:** parametric CauCA with three nonlinearities in the true mixing function.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca.csv`
  - **Used in:** Figures 4 (a), (c), (d), (e) and (g).
- **Description:** parametric CauCA with three nonlinearities in the true mixing function and only two latent variables.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-2-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_2vars.csv`
  - **Used in:** Figure 4 (c).
- **Description:** parametric CauCA with three nonlinearities in the true mixing function and only three latent variables.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-3-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_3vars.csv`
  - **Used in:** Figure 4 (c).
- **Description:** parametric CauCA with SNR 5.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --scm-coeffs-low -5 --scm-coeffs-high 5
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_5scm.csv`
  - **Used in:** Figure 4 (g).
- **Description:** parametric CauCA with three nonlinearities in the true mixing function and five latent variables.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-5-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_5vars.csv`
  - **Used in:** Figure 4 (c).
- **Description:** parametric CauCA with SNR 10.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --scm-coeffs-low -10 --scm-coeffs-high 10
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_10scm.csv`
  - **Used in:** Figure 4 (g).
- **Description:** parametric CauCA with three nonlinearities in the true mixing function with a misspecified unmixing function.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model linear --batch-size 4096 --lr 1e-3 --max-epochs 25 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --function-misspec
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_func_misspec.csv`
  - **Used in:** Figures 4 (a) and (e).
- **Description:** parametric CauCA with three nonlinearities in the true mixing function with a misspecified adjacency.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --adjacency-misspec
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_misspec.csv`
  - **Used in:** Figures 4 (a) and (e).
- **Description:** parametric CauCA with SNR 5 and misspecified adjacency.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --scm-coeffs-low -5 --scm-coeffs-high 5 --adjacency-misspec
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_misspec_5scm.csv`
  - **Used in:** Figure 4 (g).
- **Description:** parametric CauCA with SNR 10 and misspecified adjacency.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 150 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --scm-coeffs-low -10 --scm-coeffs-high 10 --adjacency-misspec
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_cauca_misspec_10scm.csv`
  - **Used in:** Figure 4 (g).
- **Description:** ICA with three nonlinearities in the true mixing function.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-8 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_ica.csv`
  - **Used in:** Figures 4 (b) and (f).
- **Description:** ICA with three nonlinearities in the true mixing function and misspecified unmixing function.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model linear --batch-size 4096 --lr 1e-3 --max-epochs 25 --k-flows 12 --dgp graph-4-8 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7 --function-misspec
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_ica_func_misspec.csv`
  - **Used in:** Figures 4 (b) and (f).
- **Description:** ICA with three nonlinearities in the true mixing function and a naive (non-multi-environment) latent model.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model naive --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-8 --n-nonlinearities 3 --lr-scheduler cosine --lr-min 1e-7
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/3_nonlin_ica_naive.csv`
  - **Used in:** Figures 4 (b) and (f).
- **Description:** parametric CauCA with five nonlinearities in the true mixing function.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-random-1 --n-nonlinearities 5 --lr-scheduler cosine --lr-min 1e-7
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/5_nonlin_cauca.csv`
  - **Used in:** Figure 4 (c).
- **Description:** nonparametric CauCA with a location-scale ground-truth SCM and a misspecified unmixing function.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model linear --scm location-scale --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-random-1 --lr-scheduler cosine --lr-min 1e-7 --nonparametric-base-distr --n-nonlinearities 3 --function-misspec
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/func_misspec.csv`
  - **Used in:** Figure 6.
- **Description:** nonparametric CauCA with a linear Gaussian ground-truth SCM.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --scm linear --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-random-1 --lr-scheduler cosine --lr-min 1e-7 --nonparametric-base-distr --n-nonlinearities 3
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/linear_scm.csv`
  - **Used in:** Figure 6.
- **Description:** nonparametric CauCA with a location-scale ground-truth SCM.
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model nonlinear --scm location-scale --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-random-1 --lr-scheduler cosine --lr-min 1e-7 --nonparametric-base-distr --n-nonlinearities 3
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/locscale.csv`
  - **Used in:** Figure 6.
- **Description:** nonparametric CauCA with a location-scale ground-truth SCM with a naive (non-multi-environment) latent model
  - **Command:**
    ```bash
    python -m experiments.cauca.main --seed <S> --training-seed <T> --model naive --scm location-scale --batch-size 4096 --lr 5e-4 --max-epochs 100 --k-flows 12 --dgp graph-4-random-1 --lr-scheduler cosine --lr-min 1e-7 --nonparametric-base-distr --n-nonlinearities 3
    ```
    where `S=0, ...,9` and `T=0, 1, 2` are used for the different seeds.
  - **Output file:** `experiments/cauca/results/data/naive.csv`
  - **Used in:** Figure 6.

</details>


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