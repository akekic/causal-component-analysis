from abc import ABC
from functools import partial
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from .utils import (
    sample_coeffs,
    linear_base_func,
    linear_inverse_jacobian,
    make_location_scale_function,
)


class MultiEnvLatentSCM(ABC):
    """
    Base class for multi-environment latent SCM.

    In environments where a variable is intervened on, the dependencies of the variable are cut. Note that this
    class only implements the causal mechanisms. The exogenous noise variables, which mayb also shift under
    interventions, are implemented in the noise generator.

    Attributes
    ----------
    adjacency_matrix : np.ndarray, shape (latent_dim, latent_dim)
        Adjacency matrix of the SCM.
    latent_dim : int
        Dimension of the latent space.
    intervention_targets_per_env : Tensor, shape (num_envs, latent_dim)
        Binary tensor indicating which variables are intervened on in each environment.
    dag : nx.DiGraph
        Directed acyclic graph representing the causal structure.
    topological_order : list[int]
        Topological order of the causal graph.
    functions_per_env : dict[int, dict[int, callable]]
        Dictionary mapping environment indices to dictionaries mapping latent variable indices to
        functions that implement the causal mechanism. I.e. functions_per_env[env][index] is a
        function that takes two arguments, v and u, and returns a Tensor of shape (batch_size,
        latent_dim) that contains the result of applying the causal mechanism the parents of index.
    inverse_jac_per_env : dict[int, dict[int, callable]]
        Dictionary mapping environment indices to dictionaries mapping latent variable indices to
        functions that implement the log of the inverse Jacobian of the causal mechanism. I.e.
        inverse_jac_per_env[env][index] is a function that takes two arguments, v and u, and
        returns a Tensor of shape (batch_size,) that contains the log of the inverse Jacobian of
        the causal mechanism applied to the parents of index.

    Methods
    -------
    push_forward(u: Tensor, env: int) -> Tensor
        Push forward the latent variable u through the SCM in environment env.
    log_inverse_jacobian(v: Tensor, u: Tensor, env: int) -> Tensor
        Compute the log of the inverse Jacobian of the SCM in environment env at v and u.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        latent_dim: int,
        intervention_targets_per_env: Tensor,
    ) -> None:
        """
        Parameters
        ----------
        adjacency_matrix: np.ndarray, shape (latent_dim, latent_dim)
        latent_dim: int
        intervention_targets_per_env: Tensor, shape (num_envs, latent_dim)
        """
        self.adjacency_matrix = adjacency_matrix
        self.latent_dim = latent_dim
        self.intervention_targets_per_env = intervention_targets_per_env
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1] == latent_dim
        self.dag = nx.DiGraph(adjacency_matrix)
        self.topological_order = list(nx.topological_sort(self.dag))

        self.functions_per_env = None
        self.inverse_jac_per_env = None

    def push_forward(self, u: Tensor, env: int) -> Tensor:
        """
        Push forward the latent variable u through the SCM in environment env.

        Parameters
        ----------
        u: Tensor, shape (num_samples, latent_dim)
            Samples of the exogenous noise variables.
        env: int
            Environment index.

        Returns
        -------
        v: Tensor, shape (num_samples, latent_dim)
            Samples of the latent variables.
        """
        v = torch.nan * torch.zeros_like(u)
        for index in self.topological_order:
            f = self.functions_per_env[env][index]
            v[:, index] = f(v, u)
        return v

    def log_inverse_jacobian(self, v: Tensor, u: Tensor, env: int) -> Tensor:
        """
        Compute the log of the inverse Jacobian of the SCM in environment env at v and u.

        Parameters
        ----------
        v: Tensor, shape (num_samples, latent_dim)
            Samples of the latent variables.
        u: Tensor, shape (num_samples, latent_dim)
            Samples of the exogenous noise variables.
        env: int
            Environment index.

        Returns
        -------
        log_inv_jac: Tensor, shape (num_samples,)
            Log of the inverse Jacobian of the SCM at v and u.
        """
        log_inv_jac = 0.0
        for index in self.topological_order:
            log_inv_jac += torch.log(self.inverse_jac_per_env[env][index](v, u))
        return log_inv_jac


class LinearSCM(MultiEnvLatentSCM):
    """
    Multi-environment latent SCM, where all causal mechanisms are linear. The coefficients of the
    linear causal mechanisms are sampled from a uniform distribution.

    Inherits all attributes and methods from MultiEnvLatentSCM.

    Additional attributes
    ---------------------
    coeffs_low : float
        Lower bound for the coefficients of the linear causal mechanisms. Default: -1.0.
    coeffs_high : float
        Upper bound for the coefficients of the linear causal mechanisms. Default: 1.0.
    coeffs_min_abs_value : Optional[float]
        Minimum absolute value for the coefficients of the linear causal mechanisms. If None, no
        minimum absolute value is enforced. Default: None.

    Additional methods
    ------------------
    setup_functions_per_env(intervention_targets_per_env: Tensor) -> tuple[dict[int, callable], dict[int, callable]]
        Set up the functions_per_env and inverse_jac_per_env attributes. This is where the linear
        causal mechanisms are defined.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        latent_dim: int,
        intervention_targets_per_env: Tensor,
        coeffs_low: float = -1.0,
        coeffs_high: float = 1.0,
        coeffs_min_abs_value: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        adjacency_matrix: np.ndarray, shape (latent_dim, latent_dim)
        latent_dim: int
        intervention_targets_per_env: Tensor, shape (num_envs, latent_dim)
        coeffs_low: float
        coeffs_high: float
        coeffs_min_abs_value: Optional[float]
        """
        super().__init__(
            adjacency_matrix,
            latent_dim,
            intervention_targets_per_env,
        )
        self.coeffs_low = coeffs_low
        self.coeffs_high = coeffs_high
        self.coeffs_min_abs_value = coeffs_min_abs_value

        base_functions = []
        base_inverse_jac = []
        base_coeff_values = []
        for index in range(self.latent_dim):
            parents = torch.tensor(
                list(self.dag.predecessors(index)), dtype=torch.int64
            )
            coeffs = sample_coeffs(
                self.coeffs_low,
                self.coeffs_high,
                (len(parents) + 1,),
                min_abs_value=self.coeffs_min_abs_value,
            )
            coeffs[-1] = 1  # set the noise coefficient to 1

            base_functions.append(
                partial(linear_base_func, index=index, parents=parents, coeffs=coeffs)
            )
            base_inverse_jac.append(
                partial(
                    linear_inverse_jacobian,
                    index=index,
                    parents=parents,
                    coeffs=coeffs,
                )
            )
            base_coeff_values.append(coeffs)
        self.base_functions = base_functions
        self.base_inverse_jac = base_inverse_jac
        self.base_coeff_values = base_coeff_values
        self.functions_per_env, self.inverse_jac_per_env = self.setup_functions_per_env(
            intervention_targets_per_env
        )

    def setup_functions_per_env(
        self, intervention_targets_per_env: Tensor
    ) -> tuple[dict[int, callable], dict[int, callable]]:
        """
        Set up the functions_per_env and inverse_jac_per_env attributes. This is where the linear
        causal mechanisms are defined.

        Parameters
        ----------
        intervention_targets_per_env: Tensor, shape (num_envs, latent_dim)
            Intervention targets for each environment.

        Returns
        -------
        functions_per_env: dict[int, dict[int, callable]]
            Dictionary mapping environment indices to dictionaries mapping latent variable indices to
            functions that implement the causal mechanism. I.e. functions_per_env[env][index] is a
            function that takes two arguments, v and u, and returns a Tensor of shape (batch_size,
            latent_dim) that contains the result of applying the causal mechanism the parents of index.
        inverse_jac_per_env: dict[int, dict[int, callable]]
            Dictionary mapping environment indices to dictionaries mapping latent variable indices to
            functions that implement the log of the inverse Jacobian of the causal mechanism. I.e.
            inverse_jac_per_env[env][index] is a function that takes two arguments, v and u, and
            returns a Tensor of shape (batch_size,) that contains the log of the inverse Jacobian of
            the causal mechanism applied to the parents of index.
        """
        functions_per_env = {}
        inverse_jac_per_env = {}
        num_envs = intervention_targets_per_env.shape[0]

        for env in range(num_envs):
            functions_env = {}
            inverse_jac_env = {}
            for index in self.topological_order:
                if intervention_targets_per_env[env][index] == 1:
                    parents = torch.tensor(
                        list(self.dag.predecessors(index)), dtype=torch.int64
                    )
                    coeffs = torch.zeros((len(parents) + 1,))  # cut edges from parents
                    coeffs[-1] = 1.0  # still use noise
                    f = partial(
                        linear_base_func,
                        index=index,
                        parents=parents,
                        coeffs=coeffs,
                    )
                    inverse_jac = partial(
                        linear_inverse_jacobian,
                        index=index,
                        parents=parents,
                        coeffs=coeffs,
                    )
                else:
                    f = self.base_functions[index]
                    inverse_jac = self.base_inverse_jac[index]
                functions_env[index] = f
                inverse_jac_env[index] = inverse_jac
            functions_per_env[env] = functions_env
            inverse_jac_per_env[env] = inverse_jac_env
        return functions_per_env, inverse_jac_per_env


class LocationScaleSCM(MultiEnvLatentSCM):
    """
    Multi-environment latent SCM, where all causal mechanisms are location-scale functions [1] of the form
    v_i = snr * f_loc(pa_i) + f_scale(u_i), where f_loc and f_scale are random nonlinear functions, pa_i
    are the parents of v_i, and u_i is the exogenous noise variable corresponding to v_i. snr is the
    signal-to-noise ratio.

    Inherits all attributes and methods from MultiEnvLatentSCM.

    Additional attributes
    ---------------------
    n_nonlinearities : int
        Number of nonlinearities in the location-scale functions. Default: 3.
    snr : float
        Signal-to-noise ratio. Default: 1.0.
    base_functions : list[callable]
        List of base functions that implement the location-scale functions for each latent variable in the
        unintervened (observational) environment.
    base_inverse_jac : list[callable]
        List of base functions that implement the log of the inverse Jacobian of the location-scale functions
        for each latent variable in the unintervened (observational) environment.

    Additional methods
    ------------------
    setup_functions_per_env(intervention_targets_per_env: Tensor) -> tuple[dict[int, callable], dict[int, callable]]
        Set up the functions_per_env and inverse_jac_per_env attributes. This is where the causal mechanisms
        based on the location-scale functions are defined.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Location%E2%80%93scale_family
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        latent_dim: int,
        intervention_targets_per_env: Tensor,
        n_nonlinearities: int = 3,
        snr: float = 1.0,
    ) -> None:
        super().__init__(
            adjacency_matrix,
            latent_dim,
            intervention_targets_per_env,
        )
        self.n_nonlinearities = n_nonlinearities
        self.snr = snr

        base_functions = []
        base_inverse_jac = []
        for index in range(self.latent_dim):
            parents = torch.tensor(
                list(self.dag.predecessors(index)), dtype=torch.int64
            )
            f, inverse_jac = make_location_scale_function(
                index, parents, n_nonlinearities, snr
            )
            base_functions.append(f)
            base_inverse_jac.append(inverse_jac)
        self.base_functions = base_functions
        self.base_inverse_jac = base_inverse_jac
        self.functions_per_env, self.inverse_jac_per_env = self.setup_functions_per_env(
            intervention_targets_per_env
        )

    def setup_functions_per_env(
        self, intervention_targets_per_env: Tensor
    ) -> tuple[dict[int, callable], dict[int, callable]]:
        functions_per_env = {}
        inverse_jac_per_env = {}
        num_envs = intervention_targets_per_env.shape[0]

        for env in range(num_envs):
            functions_env = {}
            inverse_jac_env = {}
            for index in self.topological_order:
                if intervention_targets_per_env[env][index] == 1:
                    parents = []
                    f, inverse_jac = make_location_scale_function(
                        index, parents, self.n_nonlinearities, self.snr
                    )
                else:
                    f = self.base_functions[index]
                    inverse_jac = self.base_inverse_jac[index]
                functions_env[index] = f
                inverse_jac_env[index] = inverse_jac
            functions_per_env[env] = functions_env
            inverse_jac_per_env[env] = inverse_jac_env
        return functions_per_env, inverse_jac_per_env
