import numpy as np


def sample_random_dag(n_nodes: int, edge_prob: float) -> np.ndarray:
    """
    Sample a random DAG with n_nodes nodes and edge_prob probability of an edge between two nodes.

    We ensure that there is at least one edge in the graph by rejecting graphs with no edges and
    resampling.

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the graph.
    edge_prob: float
        Probability of an edge between two nodes.

    Returns
    -------
    adjaceny_matrix: np.ndarray, shape (n_nodes, n_nodes)
        The adjacency matrix of the sampled DAG.
    """
    while True:
        adjaceny_matrix = np.random.binomial(1, edge_prob, size=(n_nodes, n_nodes))
        # put all lower triangular elements to zero
        adjaceny_matrix[np.tril_indices(n_nodes)] = 0

        # make sure the graph has at least one edge
        if np.sum(adjaceny_matrix) > 0:
            break
    return adjaceny_matrix
