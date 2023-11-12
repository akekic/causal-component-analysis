import normflows as nf


def make_spline_flows(
    K: int,
    latent_dim: int,
    net_hidden_dim: int,
    net_hidden_layers: int,
    permutation: bool = True,
) -> list[nf.flows.Flow]:
    flows = []
    for i in range(K):
        flows += [
            nf.flows.AutoregressiveRationalQuadraticSpline(
                latent_dim, net_hidden_layers, net_hidden_dim
            )
        ]
        if permutation:
            flows += [nf.flows.LULinearPermute(latent_dim)]
    return flows
