import numpy as np
import torch

DGP = {
    "graph-4-0": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 128,  # D
    },
    "graph-4-1": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 128,  # D
    },
    "graph-4-2": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 75_000,
        "observation_dim": 128,  # D
    },
    "graph-4-3": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 75_000,
        "observation_dim": 128,  # D
    },
    "graph-4-4": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 12_500,
        "observation_dim": 128,  # D
    },
    "graph-4-5": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0]]
        ),
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 75_000,
        "observation_dim": 128,  # D
    },
    "graph-4-6": {
        "num_causal_variables": 10,  # N
        "adj_matrix": np.array(
            [
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
            ]
        ),
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 75_000,
        "observation_dim": 128,  # D
    },
    "graph-4-7": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 100_000,
        "observation_dim": 4,  # D
    },
    "graph-4-8": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-9": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-10": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-9-local": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 2_000,
        "observation_dim": 4,  # D
    },
    "graph-4-random-1": {
        "num_causal_variables": 4,  # N
        "adj_matrix": None,
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-random-p000": {
        "num_causal_variables": 4,  # N
        "adj_matrix": np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        ),
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-random-p025": {
        "num_causal_variables": 4,  # N
        "adj_matrix": None,
        "edge_prob": 0.25,
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-random-p050": {
        "num_causal_variables": 4,  # N
        "adj_matrix": None,
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-random-p075": {
        "num_causal_variables": 4,  # N
        "adj_matrix": None,
        "edge_prob": 0.75,
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-random-p100": {
        "num_causal_variables": 4,  # N
        "adj_matrix": None,
        "edge_prob": 1.0,
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 4,  # D
    },
    "graph-4-random-1-local": {
        "num_causal_variables": 4,  # N
        "adj_matrix": None,
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        "num_samples_per_env": 2_000,
        "observation_dim": 4,  # D
    },
    "graph-7-random-1": {
        "num_causal_variables": 7,  # N
        "adj_matrix": None,
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 7,  # D
    },
    "graph-7-random-1-local": {
        "num_causal_variables": 7,  # N
        "adj_matrix": None,
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 2_000,
        "observation_dim": 7,  # D
    },
    "graph-2-1": {
        "num_causal_variables": 2,  # N
        "adj_matrix": np.array([[0, 1], [0, 0]]),
        "int_targets": torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 2,  # D
    },
    "graph-2-2": {
        "num_causal_variables": 2,  # N
        "adj_matrix": np.array([[0, 0], [0, 0]]),
        "int_targets": torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 2,  # D
    },
    "graph-3-1": {
        "num_causal_variables": 3,  # N
        "adj_matrix": np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]),
        "int_targets": torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 3,  # D
    },
    "graph-3-random-1": {
        "num_causal_variables": 3,  # N
        "adj_matrix": None,
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 3,  # D
    },
    "graph-5-random-1": {
        "num_causal_variables": 5,  # N
        "adj_matrix": None,
        "edge_prob": 0.5,
        "int_targets": torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
        "num_samples_per_env": 200_000,
        "observation_dim": 5,  # D
    },
}
