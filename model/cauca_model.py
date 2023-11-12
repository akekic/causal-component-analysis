from __future__ import annotations

from abc import ABC
from itertools import product
from typing import Optional, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer

from .encoder import LinearCauCAEncoder, NaiveEncoder, NonlinearCauCAEncoder
from .utils import mean_correlation_coefficient


class CauCAModel(pl.LightningModule, ABC):
    """
    Base class for Causal Component Analysis (CauCA) models. It implements the
    training loop and the evaluation metrics.

    Attributes
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    adjacency_matrix : np.ndarray, shape (num_nodes, num_nodes)
        Adjacency matrix of the causal graph assumed by the model. This is not necessarily
        the true adjacency matrix of the data generating process (see below).
    adjacency_matrix_gt : np.ndarray, shape (num_nodes, num_nodes)
        Ground truth adjacency matrix of the causal graph. This is the adjacency matrix
        of the data generating process.
    adjacency_misspecified : bool
        Whether the adjacency matrix is misspecified. If True, the model assumes a wrong
        adjacency matrix.
    lr : float
        Learning rate for the optimizer.
    weight_decay : float
        Weight decay for the optimizer.
    lr_scheduler : str
        Learning rate scheduler to use. If None, no scheduler is used. Options are
        "cosine" or None. Default: None.
    lr_min : float
        Minimum learning rate for the scheduler. Default: 0.0.
    encoder : CauCAEncoder
        The CauCA encoder. Needs to be set in subclasses.

    Methods
    -------
    training_step(batch, batch_idx) -> Tensor
        Training step.
    validation_step(batch, batch_idx) -> dict[str, Tensor]
        Validation step: basically passes data to validation_epoch_end.
    validation_epoch_end(outputs) -> None
        Computes validation metrics across all validation data.
    test_step(batch, batch_idx) -> dict[str, Tensor]
        Test step: basically passes data to test_epoch_end.
    test_epoch_end(outputs) -> None
        Computes test metrics across all test data.
    configure_optimizers() -> dict | torch.optim.Optimizer
        Configures the optimizer and learning rate scheduler.
    forward(x) -> torch.Tensor
        Computes the latent variables from the observed data.
    on_before_optimizer_step(optimizer, optimizer_idx) -> None
        Callback that is called before each optimizer step. It ensures that some gradients
        are set to zero to fix some causal mechanisms. See documentation of ParamMultiEnvCausalDistribution
        for more details.
    set_adjacency(adjacency_matrix, adjacency_misspecified) -> np.ndarray
        Sets the adjacency matrix and possibly changes it if it is misspecified.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        adjacency_misspecified: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.adjacency_matrix = self.set_adjacency(
            adjacency_matrix, adjacency_misspecified
        )
        self.adjacency_matrix_gt = adjacency_matrix
        self.adjacency_misspecified = adjacency_misspecified
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.encoder = None  # needs to be set in subclasses
        self.save_hyperparameters()

    @staticmethod
    def set_adjacency(
        adjacency_matrix: np.ndarray, adjacency_misspecified: bool
    ) -> np.ndarray:
        if not adjacency_misspecified:
            return adjacency_matrix

        if adjacency_matrix.shape[0] == 2 and np.sum(adjacency_matrix) == 0:
            # for 2 variables, if adjacency matrix is [[0, 0], [0, 0]], then
            # replace with [[0, 1], [0, 0]]

            adjacency_matrix_out = np.zeros_like(adjacency_matrix)
            adjacency_matrix_out[0, 1] = 1
            return adjacency_matrix_out
        elif adjacency_matrix.shape[0] > 2:
            raise ValueError(
                "Adjacency misspecification not supported for empty adjacency matrix for >2 variables"
            )
        else:
            return adjacency_matrix.T

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        x, v, u, e, int_target, log_prob_gt = batch
        log_prob, res = self.encoder.multi_env_log_prob(x, e, int_target)
        loss = -log_prob.mean()

        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, ...], batch_idx: int
    ) -> dict[str, Tensor]:
        x, v, u, e, int_target, log_prob_gt = batch
        log_prob, res = self.encoder.multi_env_log_prob(x, e, int_target)

        v_hat = self(x)

        return {
            "log_prob": log_prob,
            "log_prob_gt": log_prob_gt,
            "v": v,
            "v_hat": v_hat,
        }

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        log_prob = torch.cat([o["log_prob"] for o in outputs])
        log_prob_gt = torch.cat([o["log_prob_gt"] for o in outputs])

        v = torch.cat([o["v"] for o in outputs])
        v_hat = torch.cat([o["v_hat"] for o in outputs])
        mcc = mean_correlation_coefficient(v_hat, v)
        mcc_spearman = mean_correlation_coefficient(v_hat, v, method="spearman")

        loss = -log_prob.mean()
        loss_gt = -log_prob_gt.mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_loss_gt", loss_gt, prog_bar=False)
        self.log("val_mcc", mcc.mean(), prog_bar=True)
        for i, mcc_value in enumerate(mcc):
            self.log(f"val_mcc_{i}", mcc_value, prog_bar=False)
        self.log("val_mcc_spearman", mcc_spearman.mean(), prog_bar=True)
        for i, mcc_value in enumerate(mcc_spearman):
            self.log(f"val_mcc_spearman_{i}", mcc_value, prog_bar=False)

    def test_step(
        self, batch: tuple[Tensor, ...], batch_idx: int
    ) -> Union[None, dict[str, Tensor]]:
        x, v, u, e, int_target, log_prob_gt = batch
        log_prob, res = self.encoder.multi_env_log_prob(x, e, int_target)

        return {
            "log_prob": log_prob,
            "v": v,
            "v_hat": self(x),
        }

    def test_epoch_end(self, outputs: List[dict]) -> None:
        log_prob = torch.cat([o["log_prob"] for o in outputs])
        loss = -log_prob.mean()

        v = torch.cat([o["v"] for o in outputs])
        v_hat = torch.cat([o["v_hat"] for o in outputs])
        mcc = mean_correlation_coefficient(v_hat, v)

        self.log("test_loss", loss, prog_bar=False)
        self.log("test_mcc", mcc.mean(), prog_bar=False)
        for i, mcc_value in enumerate(mcc):
            self.log(f"test_mcc_{i}", mcc_value, prog_bar=False)

    def configure_optimizers(self) -> dict | torch.optim.Optimizer:
        config_dict = {}
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        config_dict["optimizer"] = optimizer

        if self.lr_scheduler == "cosine":
            # cosine learning rate annealing
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr_min,
                verbose=True,
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            }
            config_dict["lr_scheduler"] = lr_scheduler_config
        elif self.lr_scheduler is None:
            return optimizer
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        return config_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_hat = self.encoder(x)
        return v_hat

    def on_before_optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: int
    ) -> None:
        num_envs = len(self.encoder.intervention_targets_per_env)
        num_vars = self.adjacency_matrix.shape[0]

        # set gradients to fixed q0 parameters to zero
        if self.encoder.q0.trainable:
            try:
                for param_idx, (env, i) in enumerate(
                    product(range(num_envs), range(num_vars))
                ):
                    if not self.encoder.q0.noise_means_requires_grad[env][i]:
                        list(self.encoder.q0.noise_means.parameters())[
                            param_idx
                        ].grad = None
                    if not self.encoder.q0.noise_stds_requires_grad[env][i]:
                        list(self.encoder.q0.noise_stds.parameters())[
                            param_idx
                        ].grad = None
            except AttributeError:
                pass


class NonlinearCauCAModel(CauCAModel):
    """
    CauCA model with nonlinear unmixing function.

    Additional attributes
    ---------------------
    k_flows : int
        Number of flows to use in the nonlinear unmixing function. Default: 1.
    net_hidden_dim : int
        Hidden dimension of the neural network used in the nonlinear unmixing function. Default: 128.
    net_hidden_layers : int
        Number of hidden layers of the neural network used in the nonlinear unmixing function. Default: 3.
    fix_mechanisms : bool
        Some mechanisms can be fixed to a simple gaussian distribution without loss of generality.
        This has only an effect for the parametric base distribution. If True, these mechanisms are fixed.
        Default: True.
    fix_all_intervention_targets : bool
        When fixable mechanisms are fixed, this parameter determines whether all intervention targets
        are fixed (option 1) or all intervention targets which are non-root nodes together with all
        non-intervened root nodes (option 2). See documentation of ParamMultiEnvCausalDistribution
        for more details. Default: False.
    nonparametric_base_distr : bool
        Whether to use a nonparametric base distribution for the flows. If false, a parametric linear
        gaussian causal base distribution is used. Default: False.
    K_cbn : int
        Number of flows to use in the nonlinear nonparametric base distribution. Default: 3.
    net_hidden_dim_cbn : int
        Hidden dimension of the neural network used in the nonlinear nonparametric base distribution. Default: 128.
    net_hidden_layers_cbn : int
        Number of hidden layers of the neural network used in the nonlinear nonparametric base distribution. Default: 3.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        intervention_targets_per_env: Tensor,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        adjacency_misspecified: bool = False,
        k_flows: int = 1,
        net_hidden_dim: int = 128,
        net_hidden_layers: int = 3,
        fix_mechanisms: bool = True,
        fix_all_intervention_targets: bool = False,
        nonparametric_base_distr: bool = False,
        K_cbn: int = 3,
        net_hidden_dim_cbn: int = 128,
        net_hidden_layers_cbn: int = 3,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            adjacency_matrix=adjacency_matrix,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
            adjacency_misspecified=adjacency_misspecified,
        )
        self.encoder = NonlinearCauCAEncoder(
            latent_dim,
            self.adjacency_matrix,  # this is the misspecified adjacency matrix if adjacency_misspecified=True
            K=k_flows,
            intervention_targets_per_env=intervention_targets_per_env,
            net_hidden_dim=net_hidden_dim,
            net_hidden_layers=net_hidden_layers,
            fix_mechanisms=fix_mechanisms,
            fix_all_intervention_targets=fix_all_intervention_targets,
            nonparametric_base_distr=nonparametric_base_distr,
            K_cbn=K_cbn,
            net_hidden_dim_cbn=net_hidden_dim_cbn,
            net_hidden_layers_cbn=net_hidden_layers_cbn,
        )
        self.save_hyperparameters()


class LinearCauCAModel(CauCAModel):
    """
    CauCA model with linear unmixing function.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        intervention_targets_per_env: Tensor,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        adjacency_misspecified: bool = False,
        fix_mechanisms: bool = True,
        nonparametric_base_distr: bool = False,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            adjacency_matrix=adjacency_matrix,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
            adjacency_misspecified=adjacency_misspecified,
        )
        self.encoder = LinearCauCAEncoder(
            latent_dim,
            self.adjacency_matrix,  # this is the misspecified adjacency matrix if adjacency_misspecified=True
            intervention_targets_per_env=intervention_targets_per_env,
            fix_mechanisms=fix_mechanisms,
            nonparametric_base_distr=nonparametric_base_distr,
        )
        self.save_hyperparameters()


class NaiveNonlinearModel(CauCAModel):
    """
    Naive CauCA model with nonlinear unmixing function. It assumes no causal dependencies.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        lr: float = 1e-2,
        weight_decay: float = 0,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 0.0,
        adjacency_misspecified: bool = False,
        k_flows: int = 1,
        intervention_targets_per_env: Optional[torch.Tensor] = None,
        net_hidden_dim: int = 128,
        net_hidden_layers: int = 3,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            adjacency_matrix=adjacency_matrix,
            lr=lr,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            lr_min=lr_min,
            adjacency_misspecified=adjacency_misspecified,
        )
        self.encoder = NaiveEncoder(
            latent_dim,
            self.adjacency_matrix,  # this is the misspecified adjacency matrix if adjacency_misspecified=True
            K=k_flows,
            intervention_targets_per_env=intervention_targets_per_env,
            net_hidden_dim=net_hidden_dim,
            net_hidden_layers=net_hidden_layers,
        )
        self.save_hyperparameters()
