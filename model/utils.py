from itertools import permutations

import torch
from torch import Tensor
from torchmetrics import SpearmanCorrCoef


def mean_correlation_coefficient(v_hat: Tensor, v: Tensor, method="pearson") -> Tensor:
    assert v_hat.shape == v.shape

    if method == "pearson":
        # get all permutations of the columns of v_hat and v
        v_hat_permutations = list(permutations(range(v_hat.shape[1])))
        corr_coefficients_perm = torch.zeros(
            (len(v_hat_permutations), v_hat.shape[1]), device=v_hat.device
        )
        for p_idx, perm in enumerate(permutations(range(v_hat.shape[1]))):
            for i in range(v_hat.shape[1]):
                data = torch.stack([v_hat[:, i], v[:, perm[i]]], dim=1).T
                corr_coefficients_perm[p_idx, i] = torch.abs(torch.corrcoef(data)[0, 1])
        best_p_idx = torch.argmax(corr_coefficients_perm.sum(dim=1))
        corr_coefficients = corr_coefficients_perm[best_p_idx]
    elif method == "spearman":
        # Note: spearman does not check all permutations and does not work with
        # intervention target permutation
        spearman = SpearmanCorrCoef(num_outputs=v.shape[1])
        corr_coefficients = torch.abs(spearman(v_hat, v))
    else:
        raise ValueError(f"Unknown correlation coefficient method: {method}")
    return corr_coefficients
