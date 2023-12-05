import networkx as nx
import normflows as nf
import numpy as np
import torch
from torch import log, abs, Tensor
from torch.nn.functional import gaussian_nll_loss

from .utils import make_spline_flows


class MultiEnvBaseDistribution(nf.distributions.BaseDistribution):
    """
    Base distribution for nonparametric multi-environment causal distributions.

    This simple independent Gaussian distribution is used as the base distribution for the
    nonparametric multi-environment causal distribution. I.e. this distribution represents the
    exogenous noise in the SCM.
    """

    def multi_env_log_prob(
        self, x: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> Tensor:
        gaussian_nll = gaussian_nll_loss(
            x, torch.zeros_like(x), torch.ones_like(x), full=True, reduction="none"
        )
        mask = ~intervention_targets.to(torch.bool)
        log_p = -(mask * gaussian_nll).sum(dim=1)
        return log_p


class NonparamMultiEnvCausalDistribution(nf.NormalizingFlow):
    """
    Nonarametric multi-environment causal distribution.

    A nonparametric causal distribution that uses a normalizing flow to parameterize the latent
    causal mechanisms. This causal distribution has two parts:
        1. The latent SCM, which is parameterized by a normalizing flow. It represents the reduced
        form of the SCM, mapping independent (Gaussian) exogenous noise to the endogenous latent
        variables. The causal structure of the latent SCM is encoded through the topological order
        of the latent variables according to the adjacency matrix.
        2. Fixed, simple base distributions for the mechanisms that are intervened on.

    Attributes
    ----------
    adjacency_matrix : np.ndarray
        The adjacency matrix of the SCM.
    K : int
        The number of normalizing flow blocks to use for the reduced form of the SCM.
    net_hidden_dim : int
        The hidden dimension of the neural networks used in the normalizing flow blocks.
    net_hidden_layers : int
        The number of hidden layers of the neural networks used in the normalizing flow blocks.
    perm : torch.Tensor
        The permutation of the latent variables according to the topological order.

    Methods
    -------
    multi_env_log_prob(z, e, intervention_targets) -> torch.Tensor
        Compute the log probability of the given data.

    References
    ----------
    [1] https://arxiv.org/abs/2305.17225
    """

    trainable = True

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        K: int = 3,
        net_hidden_dim: int = 128,
        net_hidden_layers: int = 3,
    ) -> None:
        self.adjacency_matrix = adjacency_matrix
        self.K = K
        self.net_hidden_dim = net_hidden_dim
        self.net_hidden_layers = net_hidden_layers

        latent_dim = adjacency_matrix.shape[0]

        # permutation according to topological order
        self.perm = torch.tensor(
            list(nx.topological_sort(nx.DiGraph(self.adjacency_matrix))),
            dtype=torch.long,
        )

        flows = make_spline_flows(
            K, latent_dim, net_hidden_dim, net_hidden_layers, permutation=False
        )
        q0 = MultiEnvBaseDistribution()
        super().__init__(q0=q0, flows=flows)

    def multi_env_log_prob(
        self, z: Tensor, e: Tensor, intervention_targets: Tensor
    ) -> Tensor:
        z = z[:, self.perm]  # permute inputs to be in topological order
        log_q, u = self._determinant_terms(intervention_targets, z)
        prob_terms = self.q0.multi_env_log_prob(u, e, intervention_targets)
        prob_terms_intervened = self._prob_terms_intervened(intervention_targets, z)
        log_q += prob_terms + prob_terms_intervened

        return log_q

    def _determinant_terms(
        self, intervention_targets: Tensor, z: Tensor
    ) -> tuple[Tensor, Tensor]:
        log_q = torch.zeros(len(z), dtype=z.dtype, device=z.device)
        u = z
        for i in range(len(self.flows) - 1, -1, -1):
            u, log_det = self.flows[i].inverse(u)
            log_q += log_det

        # remove determinant terms for intervened mechanisms
        jac_row = torch.autograd.functional.jvp(
            self.inverse, z, v=intervention_targets, create_graph=True
        )[1]
        jac_diag_element = (jac_row * intervention_targets).sum(dim=1)
        # mask zero elements
        not_intervened_mask = ~intervention_targets.sum(dim=1).to(torch.bool)
        jac_diag_element[not_intervened_mask] = 1
        log_q -= log(abs(jac_diag_element) + 1e-8)
        return log_q, u

    def _prob_terms_intervened(self, intervention_targets: Tensor, z: Tensor) -> Tensor:
        """
        Compute the probability terms for the intervened mechanisms.
        """
        gaussian_nll = gaussian_nll_loss(
            z, torch.zeros_like(z), torch.ones_like(z), full=True, reduction="none"
        )
        mask = intervention_targets.to(torch.bool)
        prob_terms_intervention_targets = -(mask * gaussian_nll).sum(dim=1)
        return prob_terms_intervention_targets
