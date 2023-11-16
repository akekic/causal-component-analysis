import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

from config import DGP
from data_generator import MultiEnvDataModule, make_multi_env_dgp
from model.cauca_model import LinearCauCAModel, NaiveNonlinearModel, NonlinearCauCAModel


def int_list(arg):
    try:
        int_list = int(arg)
        return int_list
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid integer list format")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run experiment for Nonparametric Identifiability of Causal Representations from Unknown "
        "Interventions."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Accelerator to use for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Number of samples per batch.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--checkpoint-root-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint root directory.",
    )
    parser.add_argument(
        "--noise-shift-type",
        type=str,
        default="mean",
        choices=["mean", "std"],
        help="Property of noise distribution that is shifted between environments.",
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=1,
        help="Check validation loss every n epochs.",
    )
    parser.add_argument(
        "--dgp",
        type=str,
        default="graph-4-0",
        help="Data generation process to use.",
    )
    parser.add_argument(
        "--k-flows",
        type=int,
        default=1,
        help="Number of flows to use in nonlinear ICA model.",
    )
    parser.add_argument(
        "--k-flows-cbn",
        type=int,
        default=3,
        help="Number of flows to use in nonlinear latent CBN model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nonlinear",
        help="Type of encoder to use.",
        choices=["linear", "nonlinear", "naive"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--mixing",
        type=str,
        default="nonlinear",
        help="Type of mixing function to use.",
        choices=["linear", "nonlinear"],
    )
    parser.add_argument(
        "--scm",
        type=str,
        default="linear",
        help="Type of SCM to use.",
        choices=["linear", "location-scale"],
    )
    parser.add_argument(
        "--n-nonlinearities",
        type=int,
        default=1,
        help="Number of nonlinearities to use in nonlinear mixing function.",
    )
    parser.add_argument(
        "--learn-scm-params",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to learn SCM parameters.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=None,
        help="Learning rate scheduler.",
        choices=[None, "cosine"],
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine learning rate scheduler.",
    )
    parser.add_argument(
        "--scm-coeffs-low",
        type=float,
        default=-1,
        help="Lower bound for SCM coefficients.",
    )
    parser.add_argument(
        "--scm-coeffs-high",
        type=float,
        default=1,
        help="Upper bound for SCM coefficients.",
    )
    parser.add_argument(
        "--scm-coeffs-min-abs-value",
        type=float,
        default=None,
        help="Minimum absolute value for SCM coefficients.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=1.0,
        help="Signal-to-noise ratio in latent SCM.",
    )
    parser.add_argument(
        "--adjacency-misspec",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Misspecify adjacency matrix - assume ICA.",
    )
    parser.add_argument(
        "--intervention-target-misspec",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Misspecify intervention target - mix up labels and true intervention targets.",
    )
    parser.add_argument(
        "--intervention-target-perm",
        nargs="+",  # Allows multiple arguments to be passed as a list
        default=None,
        type=int_list,
        help="Permutation of intervention targets. Only used if intervention-target-misspec is True.",
    )
    parser.add_argument(
        "--net-hidden-layers",
        type=int,
        default=3,
        help="Number of hidden layers in nonlinear encoder.",
    )
    parser.add_argument(
        "--net-hidden-layers-cbn",
        type=int,
        default=3,
        help="Number of hidden layers in latent CBN model.",
    )
    parser.add_argument(
        "--net-hidden-dim",
        type=int,
        default=128,
        help="Number of hidden dimensions in nonlinear encoder.",
    )
    parser.add_argument(
        "--net-hidden-dim-cbn",
        type=int,
        default=128,
        help="Number of hidden dimensions in latent CBN model.",
    )
    parser.add_argument(
        "--fix-mechanisms",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fix fixable mechanisms in latents.",
    )
    parser.add_argument(
        "--fix-all-intervention-targets",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Fix all intervention targets.",
    )
    parser.add_argument(
        "--nonparametric-base-distr",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use nonparametric base distribution for flows.",
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to log to weights and biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="nonparam-ident",
        help="Weights & Biases project name.",
    )

    args = parser.parse_args()

    if args.wandb:
        wandb_logger = WandbLogger(project=args.wandb_project)
        wandb_logger.experiment.config.update(args, allow_val_change=True)
        checkpoint_dir = (
            Path(args.checkpoint_root_dir) / f"{wandb_logger.experiment.id}"
        )
        logger = [wandb_logger]
    else:
        checkpoint_dir = Path(args.checkpoint_root_dir) / "default"
        logger = None

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    multi_env_dgp = make_multi_env_dgp(
        latent_dim=DGP[args.dgp]["num_causal_variables"],
        observation_dim=DGP[args.dgp]["observation_dim"],
        adjacency_matrix=DGP[args.dgp]["adj_matrix"],
        intervention_targets_per_env=DGP[args.dgp]["int_targets"],
        noise_shift_type=args.noise_shift_type,
        mixing=args.mixing,
        scm=args.scm,
        n_nonlinearities=args.n_nonlinearities,
        scm_coeffs_low=args.scm_coeffs_low,
        scm_coeffs_high=args.scm_coeffs_high,
        coeffs_min_abs_value=args.scm_coeffs_min_abs_value,
        edge_prob=DGP[args.dgp].get("edge_prob", None),
        snr=args.snr,
    )
    data_module = MultiEnvDataModule(
        multi_env_dgp=multi_env_dgp,
        num_samples_per_env=DGP[args.dgp]["num_samples_per_env"],
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        intervention_targets_per_env=DGP[args.dgp]["int_targets"],
        log_dir=checkpoint_dir / "data_stats",
        intervention_target_misspec=args.intervention_target_misspec,
        intervention_target_perm=args.intervention_target_perm,
    )
    data_module.setup()

    pl.seed_everything(args.training_seed, workers=True)
    if args.intervention_target_misspec:
        # remember old intervention targets
        old_intervention_targets_per_env = DGP[args.dgp]["int_targets"]
        intervention_targets_per_env = torch.zeros_like(
            old_intervention_targets_per_env
        )

        # get target permutation from data module
        perm = data_module.intervention_target_perm

        # permute intervention targets
        for env_idx in range(intervention_targets_per_env.shape[0]):
            for i in range(intervention_targets_per_env.shape[1]):
                if old_intervention_targets_per_env[env_idx, i] == 1:
                    intervention_targets_per_env[env_idx, perm[i]] = 1
    else:
        intervention_targets_per_env = DGP[args.dgp]["int_targets"]

    # Model Initialization
    if args.model == "nonlinear":
        model = NonlinearCauCAModel(
            latent_dim=DGP[args.dgp]["num_causal_variables"],
            adjacency_matrix=data_module.medgp.adjacency_matrix,
            k_flows=args.k_flows,
            lr=args.lr,
            intervention_targets_per_env=intervention_targets_per_env,
            lr_scheduler=args.lr_scheduler,
            lr_min=args.lr_min,
            adjacency_misspecified=args.adjacency_misspec,
            net_hidden_dim=args.net_hidden_dim,
            net_hidden_layers=args.net_hidden_layers,
            fix_mechanisms=args.fix_mechanisms,
            fix_all_intervention_targets=args.fix_all_intervention_targets,
            nonparametric_base_distr=args.nonparametric_base_distr,
            K_cbn=args.k_flows_cbn,
            net_hidden_dim_cbn=args.net_hidden_dim_cbn,
            net_hidden_layers_cbn=args.net_hidden_layers_cbn,
        )
    elif args.model == "linear":
        model = LinearCauCAModel(
            latent_dim=DGP[args.dgp]["num_causal_variables"],
            adjacency_matrix=data_module.medgp.adjacency_matrix,
            lr=args.lr,
            intervention_targets_per_env=intervention_targets_per_env,
            lr_scheduler=args.lr_scheduler,
            lr_min=args.lr_min,
            adjacency_misspecified=args.adjacency_misspec,
            fix_mechanisms=args.fix_mechanisms,
            nonparametric_base_distr=args.nonparametric_base_distr,
        )
    elif args.model == "naive":
        model = NaiveNonlinearModel(
            latent_dim=DGP[args.dgp]["num_causal_variables"],
            adjacency_matrix=data_module.medgp.adjacency_matrix,
            k_flows=args.k_flows,
            lr=args.lr,
            intervention_targets_per_env=DGP[args.dgp]["int_targets"],
            lr_scheduler=args.lr_scheduler,
            lr_min=args.lr_min,
            adjacency_misspecified=args.adjacency_misspec,
            net_hidden_dim=args.net_hidden_dim,
            net_hidden_layers=args.net_hidden_layers,
        )
    else:
        raise ValueError(f"Unknown model type {args.model}")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback] if args.wandb else [],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accelerator=args.accelerator,
    )
    trainer.fit(
        model,
        datamodule=data_module,
    )
    print(f"Checkpoint dir: {checkpoint_dir}")
    trainer.test(datamodule=data_module)
