from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Uniform


def leaky_tanh(x: Tensor, alpha: float = 1.0, beta: float = 0.1) -> Tensor:
    return torch.tanh(alpha * x) + beta * x


def summary_statistics(
    x: Tensor, v: Tensor, e: Tensor, intervention_targets: Tensor
) -> dict[str, pd.DataFrame]:
    x_summary_stats = pd.DataFrame(x.numpy()).describe().T.rename_axis("index")
    v_summary_stats = pd.DataFrame(v.numpy()).describe().T.rename_axis("index")
    e_summary_stats = pd.DataFrame(e.numpy()).describe().T.rename_axis("index")
    intervention_targets_summary_stats = (
        pd.DataFrame(intervention_targets.numpy()).describe().T
    ).rename_axis("index")
    return {
        "x": x_summary_stats,
        "v": v_summary_stats,
        "e": e_summary_stats,
        "intervention_targets_per_env": intervention_targets_summary_stats,
    }


def plot_dag(adj_matrix: np.ndarray, log_dir: Path) -> None:
    G = nx.DiGraph(adj_matrix)

    fig = plt.figure()
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True, width=4.0, node_size=700, arrowsize=20)
    plt.savefig(log_dir / "dag.png")
    # release memory
    fig.clf()
    plt.close("all")


def random_perm(num_causal_variables: int) -> torch.Tensor:
    while True:
        perm = torch.randperm(num_causal_variables)
        if torch.all(perm != torch.arange(num_causal_variables)):
            return perm


def sample_invertible_matrix(n: int) -> Tensor:
    matrix = torch.rand((n, n))
    while torch.abs(torch.det(matrix)) < 0.1:
        matrix = torch.randn((n, n))
    return matrix


def sample_coeffs(
    low: float = 0.0,
    high: float = 1.0,
    size: tuple[int] = (1,),
    min_abs_value: Optional[float] = None,
) -> Tensor:
    if min_abs_value is not None:
        assert min_abs_value < max(abs(low), abs(high))
        while True:
            coeffs = Uniform(low, high).sample(size)
            if torch.all(torch.abs(coeffs) >= min_abs_value):
                return coeffs
    else:
        return Uniform(low, high).sample(size)


def linear_base_func(
    v: Tensor, u: Tensor, index: int, parents: Tensor, coeffs: Tensor
) -> Tensor:
    assert len(parents) + 1 == len(coeffs)

    if len(parents) == 0:
        return coeffs * u[:, index]
    else:
        vec = torch.concatenate((v[:, parents].T, u[:, index].unsqueeze(0)), dim=0)
        return coeffs.matmul(vec)


def linear_inverse_jacobian(
    v: Tensor, u: Tensor, index: int, parents: Tensor, coeffs: Tensor
) -> Tensor:
    assert len(parents) + 1 == len(coeffs)

    if len(parents) == 0:
        return torch.ones_like(u[:, index])
    else:
        return torch.abs(coeffs[-1] * torch.ones_like(u[:, index]))


def sample_random_matrix(*size: int) -> Tensor:
    return torch.randn(*size)


def make_random_nonlinear_func(
    input_dim: int, output_dim: int, n_nonlinearities: int
) -> callable:
    assert input_dim > 0, "input_dim must be positive"
    assert output_dim > 0, "output_dim must be positive"
    assert n_nonlinearities > 0, "must have at least one nonlinearity"

    matrices = []
    for i in range(n_nonlinearities - 1):
        matrices.append(sample_random_matrix(input_dim, input_dim))
    matrices.append(sample_random_matrix(output_dim, input_dim))

    nonlinearities = []
    for i in range(n_nonlinearities):
        nonlinearities.append(leaky_tanh)

    def nonlinear_func(input: Tensor) -> Tensor:
        output = input.T
        for i in range(n_nonlinearities):
            output = nonlinearities[i](matrices[i].matmul(output))
        return output.T

    return nonlinear_func


def make_location_scale_function(
    index: int,
    parents: Union[list[int], Tensor],
    n_nonlinearities: int,
    snr: float = 1.0,
) -> tuple[callable, callable]:
    if len(parents) == 0:
        return lambda v, u: u[:, index], lambda v, u: torch.ones_like(u[:, index])

    loc_func = make_random_nonlinear_func(len(parents), 1, n_nonlinearities)
    scale_func = make_random_nonlinear_func(len(parents), 1, n_nonlinearities)

    def location_scale_func(v, u):
        loc = loc_func(v[:, parents])
        scale = scale_func(v[:, parents])
        return (snr * loc + scale * u[:, index].unsqueeze(1)).squeeze(1)

    def inverse_jacobian(v, u):
        return torch.abs(1.0 / scale_func(v[:, parents])).squeeze(1)

    return location_scale_func, inverse_jacobian
