from abc import ABC
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from .utils import leaky_tanh, sample_invertible_matrix


class MixingFunction(ABC):
    """
    Base class for mixing functions.

    The mixing function is the function that maps from the latent space to the observation space.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space.
    observation_dim: int
        Dimension of the observation space.
    """

    def __init__(self, latent_dim: int, observation_dim: int) -> None:
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim

    def __call__(self, v: Tensor) -> Tensor:
        """
        Apply the mixing function to the latent variables.

        Parameters
        ----------
        v: Tensor, shape (num_samples, latent_dim)
            Latent variables.

        Returns
        -------
        x: Tensor, shape (num_samples, observation_dim)
            Observed variables.
        """
        raise NotImplementedError()

    def save_coeffs(self, path: Path) -> None:
        """
        Save the coefficients of the mixing function to disk.

        Parameters
        ----------
        path: Path
            Path to save the coefficients to.
        """
        raise NotImplementedError()

    def unmixing_jacobian(self, v: Tensor) -> Tensor:
        """
        Compute the jacobian of the inverse mixing function using autograd and the inverse function theorem.

        Parameters
        ----------
        v: Tensor, shape (num_samples, latent_dim)
            Latent variables.

        Returns
        -------
        unmixing_jacobian: Tensor, shape (num_samples, observation_dim, latent_dim)
            Jacobian of the inverse mixing function.

        References
        ----------
        https://en.wikipedia.org/wiki/Inverse_function_theorem
        https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771/7
        """
        func = self.__call__
        inputs = v

        mixing_jacobian = torch.vmap(torch.func.jacrev(func))(inputs)
        unmixing_jacobian = torch.inverse(mixing_jacobian)

        return unmixing_jacobian


class LinearMixing(MixingFunction):
    """
    Linear mixing function. The coefficients are sampled from a uniform distribution.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space.
    observation_dim: int
        Dimension of the observation space.
    """

    def __init__(self, latent_dim: int, observation_dim: int) -> None:
        super().__init__(latent_dim, observation_dim)
        self.coeffs = torch.rand((latent_dim, observation_dim))

    def __call__(self, v: Tensor) -> Tensor:
        return torch.matmul(v, self.coeffs.to(v.device))

    def save_coeffs(self, path: Path) -> None:
        # save matrix coefficients
        torch.save(self.coeffs, path / "matrix.pt")
        matrix_np = self.coeffs.numpy()  # convert to Numpy array
        df = pd.DataFrame(matrix_np)  # convert to a dataframe
        df.to_csv(path / "matrix.csv", index=False)  # save as csv


class NonlinearMixing(MixingFunction):
    """
    Nonlinear mixing function.

    The function is composed of a number of invertible matrices and leaky-tanh nonlinearities. I.e. we
    apply a random neural network to the latent variables.

    Parameters
    ----------
    latent_dim: int
        Dimension of the latent space.
    observation_dim: int
        Dimension of the observation space.
    n_nonlinearities: int
        Number of layers (i.e. invertible maps and nonlinearities) in the mixing function. Default: 1.
    """

    def __init__(
        self, latent_dim: int, observation_dim: int, n_nonlinearities: int = 1
    ) -> None:
        super().__init__(latent_dim, observation_dim)
        assert latent_dim == observation_dim
        self.coefs = torch.rand((latent_dim, observation_dim))
        self.n_nonlinearities = n_nonlinearities

        matrices = []
        for i in range(n_nonlinearities):
            matrices.append(sample_invertible_matrix(observation_dim))
        self.matrices = matrices

        nonlinearities = []
        for i in range(n_nonlinearities):
            nonlinearities.append(leaky_tanh)
        self.nonlinearities = nonlinearities

    def __call__(self, v: Tensor) -> Tensor:
        x = v
        for i in range(self.n_nonlinearities):
            mat = self.matrices[i].to(v.device)
            nonlinearity = self.nonlinearities[i]
            x = nonlinearity(torch.matmul(x, mat))
        return x

    def save_coeffs(self, path: Path) -> None:
        # save matrix coefficients
        for i in range(self.n_nonlinearities):
            torch.save(self.matrices[i], path / f"matrix_{i}.pt")
            matrix_np = self.matrices[i].numpy()  # convert to Numpy array
            df = pd.DataFrame(matrix_np)  # convert to a dataframe
            df.to_csv(path / f"matrix_{i}.csv", index=False)  # save as csv

        # save matrix determinants in one csv
        matrix_determinants = []
        for i in range(self.n_nonlinearities):
            matrix_determinants.append(torch.det(self.matrices[i]))
        matrix_determinants_np = torch.stack(matrix_determinants).numpy()
        df = pd.DataFrame(matrix_determinants_np)
        df.to_csv(path / "matrix_determinants.csv")
