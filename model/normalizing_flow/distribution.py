from abc import ABC

import networkx as nx
import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from normflows.distributions import DiagGaussian
from torch import Tensor
from torch.distributions import Uniform
from torch.nn import ParameterList


class MultiEnvCausalDistribution(nf.distributions.BaseDistribution, ABC):
    """
    Base class for parametric multi-environment causal distributions.

    In typical normalizing flow architectures, the base distribution is a simple distribution
    such as a multivariate Gaussian. In our case, the base distribution has additional multi-environment
    causal structure. Hence, in the parametric case, this class learns the parameters of the causal
    mechanisms and noise distributions. The causal graph is assumed to be known.

    This is a subclass of BaseDistribution, which is a subclass of torch.nn.Module. Hence, this class
    can be used as a base distribution in a normalizing flow.

    Methods
    -------
    multi_env_log_prob(z, e, intervention_targets) -> Tensor
        Compute the log probability of the latent variables v in environment e, given the intervention targets.
        This is used as the main training objective.
    """

    def multi_env_log_prob(
        self, z: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> Tensor:
        raise NotImplementedError


class ParamMultiEnvCausalDistribution(MultiEnvCausalDistribution):
    """
    Parametric multi-environment causal distribution.

    This class learns the parameters of the causal mechanisms and noise distributions. The causal mechanisms
    are assumed to be linear, and the noise distributions are assumed to be Gaussian. In environments where
    a variable is intervened on, the connection to its parents is assumed to be cut off, and the noise distribution
    can be shifted relative to the observational environment (when the variable is not intervened on).

    Theoretically, we can fix some of the mechanisms involved w.l.o.g. and still achieve identifiability
    (see Appendix G2 of [1]). There are two ways to do this:
        1. Fix all mechanisms that are intervened on.
        2. Fix all observational mechanisms with an empty parent set and all intervened mechanisms with a
        non-empty parent set.
    However, we do not have to fix any of the mechanisms and in practice, we find that this leads to better
    performance.

    Attributes
    ----------
    adjacency_matrix: np.ndarray
        Adjacency matrix of the causal graph.
    fix_mechanisms: bool
        Whether to fix any of the mechanisms. Default: False.
    fix_all_intervention_targets: bool
        Whether to fix all mechanisms that are intervened on (option 1 above). If False, we fix all observational
        mechanisms with an empty parent set and all intervened mechanisms with a non-empty parent set (option 2 above).
        Default: False.
    intervention_targets_per_env: Tensor, shape (num_envs, latent_dim)
        Intervention targets per environment, with 1 indicating that the variable is intervened on
        and 0 indicating that the variable is not intervened on. This variable also implicitly defines
        the number of environments.
    dag: nx.DiGraph
        Directed acyclic graph of the causal connections.
    coeff_values: nn.ParameterList
        List of lists of coefficients for the linear mechanisms. The outer list has length equal to the number of
        variables, and the inner list has length equal to the number of parents of the variable. The last element
        of the inner list is the variance parameter. I.e. coeff_values[i][:-1] are the linear weights of the parent
        variables of variable i, and coeff_values[i][-1] is weight of the exogenous noise.
    noise_means: nn.ParameterList
        List of lists of means for the noise distributions. The outer list has length equal to the number of
        environments, and the inner list has length equal to the number of variables. noise_means[e][i] is the mean
        of the noise distribution for variable i in environment e. Note that not all of these parameters are
        used in the computation of the log probability. If a variable i is not intervened on in environment e,
        we use the observational noise distribution, i.e. noise_means[0][i] (e=0 is assumed to be the
        observational environment).
    noise_stds: nn.ParameterList
        Same as noise_means, but for the standard deviations of the noise distributions.
    coeff_values_requires_grad: list[list[bool]]
        Whether each coefficient is trainable. This is used to fix the coefficients of the mechanisms.
    noise_means_requires_grad: list[list[bool]]
        Whether each noise mean is trainable. This is used to fix the noise means of the mechanisms.
    noise_stds_requires_grad: list[list[bool]]
        Whether each noise standard deviation is trainable. This is used to fix the noise standard deviations

    References
    ----------
    [1] https://arxiv.org/abs/2305.17225
    """

    trainable = True

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        intervention_targets_per_env: Tensor,
        fix_mechanisms: bool = False,
        fix_all_intervention_targets: bool = False,
    ) -> None:
        super().__init__()
        self.adjacency_matrix = adjacency_matrix
        self.fix_mechanisms = fix_mechanisms
        self.fix_all_intervention_targets = fix_all_intervention_targets
        self.intervention_targets_per_env = intervention_targets_per_env

        self.dag = nx.DiGraph(adjacency_matrix)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        coeff_values, coeff_values_requires_grad = self._set_initial_coeffs(
            self.dag, device
        )
        noise_means, noise_means_requires_grad = self._set_initial_noise_means(
            self.dag,
            fix_mechanisms,
            intervention_targets_per_env,
            fix_all_intervention_targets,
            device,
        )
        noise_stds, noise_stds_requires_grad = self._set_initial_noise_stds(
            self.dag,
            fix_mechanisms,
            intervention_targets_per_env,
            fix_all_intervention_targets,
            device,
        )

        self.coeff_values = nn.ParameterList(coeff_values)
        self.noise_means = nn.ParameterList(noise_means)
        self.noise_stds = nn.ParameterList(noise_stds)

        self.coeff_values_requires_grad = coeff_values_requires_grad
        self.noise_means_requires_grad = noise_means_requires_grad
        self.noise_stds_requires_grad = noise_stds_requires_grad

    def multi_env_log_prob(
        self, z: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> Tensor:
        log_p = torch.zeros(len(z), dtype=z.dtype, device=z.device)
        for env in e.unique():
            env_mask = (e == env).flatten()
            z_env = z[env_mask, :]
            intervention_targets_env = intervention_targets[env_mask, :]

            for i in range(z.shape[1]):
                parents = list(self.dag.predecessors(i))

                if len(parents) == 0 or intervention_targets_env[0, i] == 1:
                    parent_contribution = 0
                else:
                    coeffs_raw = self.coeff_values[i][:-1]
                    if isinstance(coeffs_raw, nn.ParameterList):
                        coeffs_raw = torch.cat([c for c in coeffs_raw])
                    parent_coeffs = coeffs_raw.to(z.device)
                    parent_contribution = parent_coeffs.matmul(z_env[:, parents].T)

                noise_env_idx = int(env) if intervention_targets_env[0, i] == 1 else 0
                var = self.noise_stds[noise_env_idx][i] ** 2 * torch.ones_like(
                    z_env[:, i]
                )
                noise_coeff = self.coeff_values[i][-1].to(z.device)
                noise_contribution = noise_coeff * self.noise_means[noise_env_idx][i]
                var *= noise_coeff ** 2

                log_p[env_mask] += torch.distributions.Normal(
                    parent_contribution + noise_contribution, var.sqrt()
                ).log_prob(z_env[:, i])

        return log_p

    @staticmethod
    def _set_initial_coeffs(
        dag: nx.DiGraph, device: torch.device
    ) -> tuple[list[ParameterList], list[list[bool]]]:
        coeff_values = []
        coeff_values_requires_grad = []
        for i in range(dag.number_of_nodes()):
            coeff_values_i = []
            coeff_values_requires_grad_i = []
            num_parents = len(list(dag.predecessors(i)))
            for j in range(num_parents):
                random_val = Uniform(-1, 1).sample((1,))
                val = random_val
                param = nn.Parameter(val * torch.ones(1), requires_grad=True).to(device)
                coeff_values_i.append(param)
                coeff_values_requires_grad_i.append(True)
            const = torch.ones(1, requires_grad=False).to(device)  # variance param
            coeff_values_i.append(const)
            coeff_values_requires_grad_i.append(False)
            coeff_values.append(nn.ParameterList(coeff_values_i))
            coeff_values_requires_grad.append(coeff_values_requires_grad_i)
        return coeff_values, coeff_values_requires_grad

    @staticmethod
    def _set_initial_noise_means(
        dag: nx.DiGraph,
        fix_mechanisms: bool,
        intervention_targets_per_env: Tensor,
        fix_all_intervention_targets: bool,
        device: torch.device,
    ) -> tuple[list[ParameterList], list[list[bool]]]:
        noise_means = []
        noise_means_requires_grad = []
        num_envs = intervention_targets_per_env.shape[0]

        for e in range(num_envs):
            noise_means_e = []
            noise_means_requires_grad_e = []
            for i in range(dag.number_of_nodes()):
                is_shifted = intervention_targets_per_env[e][i] == 1
                is_root = len(list(dag.predecessors(i))) == 0
                if fix_all_intervention_targets:
                    is_fixed = is_shifted
                else:
                    is_fixed = (is_shifted and not is_root) or (
                        not is_shifted and is_root
                    )
                is_fixed = is_fixed and fix_mechanisms
                random_val = Uniform(-0.5, 0.5).sample((1,))
                val = random_val
                param = (
                    nn.Parameter(val * torch.ones(1), requires_grad=not is_fixed)
                ).to(device)
                noise_means_e.append(param)
                noise_means_requires_grad_e.append(not is_fixed)
            noise_means.append(nn.ParameterList(noise_means_e))
            noise_means_requires_grad.append(noise_means_requires_grad_e)
        return noise_means, noise_means_requires_grad

    @staticmethod
    def _set_initial_noise_stds(
        dag: nx.DiGraph,
        fix_mechanisms: bool,
        intervention_targets_per_env: Tensor,
        fix_all_intervention_targets: bool,
        device: torch.device,
    ) -> tuple[list[ParameterList], list[list[bool]]]:
        noise_stds = []
        noise_stds_requires_grad = []
        for e in range(intervention_targets_per_env.shape[0]):
            noise_stds_e = []
            noise_stds_requires_grad_e = []
            for i in range(dag.number_of_nodes()):
                is_shifted = intervention_targets_per_env[e][i] == 1
                is_root = len(list(dag.predecessors(i))) == 0
                if fix_all_intervention_targets:
                    is_fixed = is_shifted
                else:
                    is_fixed = (is_shifted and not is_root) or (
                        not is_shifted and is_root
                    )
                is_fixed = is_fixed and fix_mechanisms
                random_val = Uniform(0.5, 1.5).sample((1,))
                val = random_val
                param = (
                    nn.Parameter(val * torch.ones(1), requires_grad=not is_fixed)
                ).to(device)
                noise_stds_e.append(param)
                noise_stds_requires_grad_e.append(not is_fixed)
            noise_stds.append(nn.ParameterList(noise_stds_e))
            noise_stds_requires_grad.append(noise_stds_requires_grad_e)
        return noise_stds, noise_stds_requires_grad


class NaiveMultiEnvCausalDistribution(MultiEnvCausalDistribution):
    """
    Naive multi-environment causal distribution.

    This is a dummy-version of ParamMultiEnvCausalDistribution, where the causal mechanisms are assumed to
    be trivial (no connectioons between variables) and the noise distributions are assumed to be Gaussian
    and independent of the environment. This is equivalent to the independent component analysis (ICA) case.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
    ) -> None:
        super().__init__()
        self.adjacency_matrix = adjacency_matrix

        self.q0 = DiagGaussian(adjacency_matrix.shape[0], trainable=True)

    def multi_env_log_prob(
        self, z: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> Tensor:
        return self.q0.log_prob(z)
