from abc import ABC

import torch
from torch import Tensor
from torch.distributions import Uniform


class MultiEnvNoise(ABC):
    """
    Base class for multi-environment noise generators.

    Attributes
    ----------
    latent_dim: int
        Latent dimension.
    intervention_targets_per_env: Tensor, shape (num_envs, latent_dim)
        Intervention targets per environment, with 1 indicating that the variable is intervened on
        and 0 indicating that the variable is not intervened on. This variable also implicitly defines
        the number of environments.
    mean: float
        Mean of the noise distribution. If shift is True and shift_type is "mean", the mean of the noise
        distribution is shifted up or down depending on whether the mechanism is intervened on or not. Default: 0.0.
    std: float
        Standard deviation of the noise distribution. If shift is True and shift_type is "std", the standard
        deviation of the noise distribution is shifted up or down depending on whether the mechanism is intervened
        on or not. Default: 1.0.
    shift: bool
        Whether to shift the noise distribution for variables that are intervened on. Default: False.
    shift_type: str
        Whether to shift the mean or standard deviation of the noise distribution for variables that are intervened on.
        Options: "mean" or "std". Default: "mean".

    Methods
    -------
    sample(e, size=1) -> Tensor
        Sample from the noise distribution for a given environment.
    """

    def __init__(
        self,
        latent_dim: int,
        intervention_targets_per_env: Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        shift: bool = False,
        shift_type: str = "mean",
    ) -> None:
        self.latent_dim = latent_dim
        self.intervention_targets = intervention_targets_per_env
        self.mean = mean
        self.std = std
        self.shift = shift
        assert shift_type in ["mean", "std"], f"Invalid shift type: {shift_type}"
        self.shift_type = shift_type

    def sample(self, e: int, size: int = 1) -> Tensor:
        """
        Sample from the noise distribution for a given environment.

        Parameters
        ----------
        e: int
            Environment index. Must be in {0, ..., num_envs-1}. The number of environments is implicitly defined
            by the intervention_targets_per_env variable.
        size: int
            Number of samples to generate. Default: 1.

        Returns
        -------
        Tensor, shape (size, latent_dim)
            Samples from the noise distribution.
        """
        raise NotImplementedError()

    def log_prob(self, u: Tensor, e: int) -> Tensor:
        """
        Compute the log probability of u under the noise distribution for a given environment. We assume
        that all samples come from the same environment.

        Parameters
        ----------
        u: Tensor, shape (size, latent_dim)
            Samples from the noise distribution.
        e: int
            Environment index. Must be in {0, ..., num_envs-1}. The number of environments is implicitly defined
            by the intervention_targets_per_env variable.

        Returns
        -------
        log_prob: Tensor, shape (size, latent_dim)
            Log probability of u.
        """
        raise NotImplementedError()


class GaussianNoise(MultiEnvNoise):
    def __init__(
        self,
        latent_dim: int,
        intervention_targets_per_env: Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        shift: bool = False,
        shift_type: str = "mean",
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            intervention_targets_per_env=intervention_targets_per_env,
            mean=mean,
            std=std,
            shift=shift,
            shift_type=shift_type,
        )
        self.means_per_env, self.stds_per_env = self.setup_params(
            intervention_targets_per_env
        )

    def setup_params(
        self, intervention_targets_per_env: Tensor
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        means_per_env = {}
        stds_per_env = {}
        for e in range(intervention_targets_per_env.shape[0]):
            if self.shift_type == "mean":
                stds = (
                    torch.ones(self.latent_dim) * self.std
                )  # stds_per_env per dimension
                stds_per_env[e] = stds

                means = (
                    torch.ones(self.latent_dim) * self.mean
                )  # means_per_env per dimension

                # shift mean up or down if mechanism is intervened on
                if intervention_targets_per_env is not None and self.shift:
                    for i in range(self.latent_dim):
                        if intervention_targets_per_env[e][i] == 1:
                            coin_flip = torch.randint(0, 2, (1,)).item()  # 0 or 1
                            factor = 2
                            means[i] = (
                                self.mean
                                + coin_flip * factor * self.std
                                + (1 - coin_flip) * factor * self.std
                            )

                    means_per_env[e] = means

            elif self.shift_type == "std":
                means = (
                    torch.ones(self.latent_dim) * self.mean
                )  # means_per_env per dimension
                means_per_env[e] = means
                stds = (
                    torch.ones(self.latent_dim) * self.std
                )  # stds_per_env per dimension

                # shift std up or down if mechanism is intervened on
                if intervention_targets_per_env is not None and self.shift:
                    for i in range(self.latent_dim):
                        if intervention_targets_per_env[e][i] == 1:
                            coin_flip = torch.randint(0, 2, (1,)).item()  # 0 or 1
                            std_scaling_factor = (
                                Uniform(0.25, 0.75).sample(torch.Size((1,)))
                                if coin_flip == 0
                                else Uniform(1.25, 1.75).sample(torch.Size((1,)))
                            )
                            stds[i] = stds[i] * std_scaling_factor
                stds_per_env[e] = stds
            else:
                raise ValueError(f"Invalid shift type: {self.shift_type}")

        return means_per_env, stds_per_env

    def sample(self, e: int, size: int = 1) -> Tensor:
        return torch.normal(
            self.means_per_env[e].unsqueeze(0).repeat(size, 1),
            self.stds_per_env[e].unsqueeze(0).repeat(size, 1),
        )

    def log_prob(self, u: Tensor, e: int) -> Tensor:
        return torch.distributions.Normal(
            self.means_per_env[e].unsqueeze(0).repeat(u.shape[0], 1),
            self.stds_per_env[e].unsqueeze(0).repeat(u.shape[0], 1),
        ).log_prob(u)
