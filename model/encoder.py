from typing import Optional

import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from torch import abs, det, log, Tensor

from .normalizing_flow import ParamMultiEnvCausalDistribution
from .normalizing_flow.distribution import NaiveMultiEnvCausalDistribution
from .normalizing_flow.nonparametric_distribution import (
    NonparamMultiEnvCausalDistribution,
)
from .normalizing_flow.utils import make_spline_flows


class CauCAEncoder(nf.NormalizingFlow):
    """
    CauCA encoder for multi-environment data.

    The encoder maps from the observed data x to the latent space v_hat. The latent space is
    assumed to have causal structure. The encoder is trained to maximize the likelihood of
    the data under the causal model. x and v_hat are assumed to have the same dimension.

    The encoder has two main components:
        1. A causal base distribution q0 over the latent space. This encodes the latent
        causal structure.
        2. An unmixing function mapping from the observations to the latent space.

    Attributes
    ----------
    latent_dim: int
        Dimension of the latent and observed variables.
    adjacency_matrix: np.ndarray, shape (latent_dim, latent_dim)
        Adjacency matrix of the latent causal graph.
    intervention_targets_per_env: Tensor, shape (no_envs, latent_dim)
        Which variables are intervened on in each environment.
    fix_mechanisms: bool
        Whether to fix some fixable mechanisms in the causal model. (See documentation of the
        ParamMultiEnvCausalDistribution for details.) Default: False.
    fix_all_intervention_targets: bool
        Whether to fix all intervention targets in the causal model. (See documentation of the
        ParamMultiEnvCausalDistribution for details.) Default: False.
    nonparametric_base_distr: bool
        Whether to use a nonparametric base distribution. If False, a parametric base distribution
        assuming linear causal mechanisms is used. Default: False.
    flows: Optional[list[nf.flows.Flow]]
        List of normalizing flows to use for the unmixing function. Default: None.
    q0: Optional[nf.distributions.BaseDistribution]
        Base distribution over the latent space. Default: None.
    K_cbn: int
        Number of normalizing flows to use for the nonparametric base distribution. Default: 3.
    net_hidden_dim_cbn: int
        Hidden dimension of the neural network used in the nonparametric base distribution. Default: 128.
    net_hidden_layers_cbn: int
        Number of hidden layers in the neural network used in the nonparametric base distribution. Default: 3.

    Methods
    -------
    multi_env_log_prob(x, e, intervention_targets) -> Tensor
        Computes log probability of x in environment e.
    forward(x) -> Tensor
        Maps from the observed data x to the latent space v_hat.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        intervention_targets_per_env: Optional[Tensor] = None,
        fix_mechanisms: bool = False,
        fix_all_intervention_targets: bool = False,
        nonparametric_base_distr: bool = False,
        flows: Optional[list[nf.flows.Flow]] = None,
        q0: Optional[nf.distributions.BaseDistribution] = None,
        K_cbn: int = 3,
        net_hidden_dim_cbn: int = 128,
        net_hidden_layers_cbn: int = 3,
    ) -> None:
        self.latent_dim = latent_dim
        self.adjacency_matrix = adjacency_matrix
        self.intervention_targets_per_env = intervention_targets_per_env
        self.fix_mechanisms = fix_mechanisms
        self.fix_all_intervention_targets = fix_all_intervention_targets
        self.nonparametric_base_distr = nonparametric_base_distr
        self.K_cbn = K_cbn
        self.net_hidden_dim_cbn = net_hidden_dim_cbn
        self.net_hidden_layers_cbn = net_hidden_layers_cbn

        if q0 is None:
            if self.nonparametric_base_distr:
                q0 = NonparamMultiEnvCausalDistribution(
                    adjacency_matrix=adjacency_matrix,
                    K=K_cbn,
                    net_hidden_dim=net_hidden_dim_cbn,
                    net_hidden_layers=net_hidden_layers_cbn,
                )
            else:
                assert (
                    intervention_targets_per_env is not None
                ), "intervention_targets_per_env must be provided for parametric base distribution"
                q0 = ParamMultiEnvCausalDistribution(
                    adjacency_matrix=adjacency_matrix,
                    intervention_targets_per_env=intervention_targets_per_env,
                    fix_mechanisms=fix_mechanisms,
                    fix_all_intervention_targets=fix_all_intervention_targets,
                )
        super().__init__(q0=q0, flows=flows if flows is not None else [])

    def multi_env_log_prob(
        self, x: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class LinearCauCAEncoder(CauCAEncoder):
    """
    Linear CauCA encoder for multi-environment data.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        intervention_targets_per_env: Optional[Tensor] = None,
        fix_mechanisms: bool = True,
        fix_all_intervention_targets: bool = False,
        nonparametric_base_distr: bool = False,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            adjacency_matrix=adjacency_matrix,
            intervention_targets_per_env=intervention_targets_per_env,
            fix_mechanisms=fix_mechanisms,
            fix_all_intervention_targets=fix_all_intervention_targets,
            nonparametric_base_distr=nonparametric_base_distr,
        )
        self.unmixing = nn.Linear(latent_dim, latent_dim, bias=False)

    def multi_env_log_prob(
        self, x: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        v_hat = self(x)
        jacobian = torch.autograd.functional.jacobian(
            self.unmixing, x[0, :], create_graph=True
        )

        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        log_q += log(abs(det(jacobian)))
        determinant_terms = log_q

        prob_terms = self.q0.multi_env_log_prob(v_hat, e, intervention_targets)
        log_q += prob_terms
        res = {
            "log_prob": log_q,
            "determinant_terms": determinant_terms,
            "prob_terms": prob_terms,
        }
        return log_q, res

    def forward(self, x: Tensor) -> Tensor:
        latent = self.unmixing(x)
        return latent


class NonlinearCauCAEncoder(CauCAEncoder):
    """
    Nonlinear CauCA encoder for multi-environment data.

    Here the unmixing function is a normalizing flow.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent and observed variables.
    adjacency_matrix: np.ndarray, shape (latent_dim, latent_dim)
        Adjacency matrix of the latent causal graph.
    K: int
        Number of normalizing flows to use for the unmixing function. Default: 1.
    intervention_targets_per_env: Tensor, shape (no_envs, latent_dim)
        Which variables are intervened on in each environment.
    net_hidden_dim: int
        Hidden dimension of the neural network used in the normalizing flows. Default: 128.
    net_hidden_layers: int
        Number of hidden layers in the neural network used in the normalizing flows. Default: 3.
    q0: Optional[nf.distributions.BaseDistribution]
        Base distribution over the latent space. Default: None.
    K_cbn: int
        Number of normalizing flows to use for the nonparametric base distribution. Default: 3.
    net_hidden_dim_cbn: int
        Hidden dimension of the neural network used in the nonparametric base distribution. Default: 128.
    net_hidden_layers_cbn: int
        Number of hidden layers in the neural network used in the nonparametric base distribution. Default: 3.
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        K: int = 1,
        intervention_targets_per_env: Optional[Tensor] = None,
        net_hidden_dim: int = 128,
        net_hidden_layers: int = 3,
        fix_mechanisms: bool = True,
        fix_all_intervention_targets: bool = False,
        nonparametric_base_distr: bool = False,
        q0: Optional[nf.distributions.BaseDistribution] = None,
        K_cbn: int = 3,
        net_hidden_dim_cbn: int = 128,
        net_hidden_layers_cbn: int = 3,
    ) -> None:
        self.K = K
        self.intervention_targets_per_env = intervention_targets_per_env
        self.net_hidden_dim = net_hidden_dim
        self.net_hidden_layers = net_hidden_layers

        flows = make_spline_flows(K, latent_dim, net_hidden_dim, net_hidden_layers)

        super().__init__(
            latent_dim=latent_dim,
            adjacency_matrix=adjacency_matrix,
            intervention_targets_per_env=intervention_targets_per_env,
            fix_mechanisms=fix_mechanisms,
            fix_all_intervention_targets=fix_all_intervention_targets,
            nonparametric_base_distr=nonparametric_base_distr,
            flows=flows,
            q0=q0,
            K_cbn=K_cbn,
            net_hidden_dim_cbn=net_hidden_dim_cbn,
            net_hidden_layers_cbn=net_hidden_layers_cbn,
        )

    def multi_env_log_prob(
        self, x: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        determinant_terms = log_q
        prob_terms = self.q0.multi_env_log_prob(z, e, intervention_targets)
        log_q += prob_terms
        res = {
            "log_prob": log_q,
            "determinant_terms": determinant_terms,
            "prob_terms": prob_terms,
        }
        return log_q, res

    def forward(self, x: Tensor) -> Tensor:
        return self.inverse(x)


class NaiveEncoder(NonlinearCauCAEncoder):
    """
    Naive encoder for multi-environment data.

    This encoder does not assume any causal structure in the latent space. Equivalent to independent
    components analysis (ICA).
    """

    def __init__(
        self,
        latent_dim: int,
        adjacency_matrix: np.ndarray,
        K: int = 1,
        intervention_targets_per_env: Optional[Tensor] = None,
        net_hidden_dim: int = 128,
        net_hidden_layers: int = 3,
    ) -> None:
        # overwrite the q0 from NonlinearICAEncoder
        q0 = NaiveMultiEnvCausalDistribution(
            adjacency_matrix=adjacency_matrix,
        )

        super().__init__(
            latent_dim=latent_dim,
            adjacency_matrix=adjacency_matrix,
            K=K,
            intervention_targets_per_env=intervention_targets_per_env,
            net_hidden_dim=net_hidden_dim,
            net_hidden_layers=net_hidden_layers,
            q0=q0,
        )
