import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear
import numpy as np
from torch.nn.functional import softmax

"""
    SE(3)-Transformer, code referenced from:
        1). https://docs.e3nn.org/en/latest/guide/transformer.html
        2). S. Batzner, A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky, E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials, Nat Commun 13, 2453 (2022).
            https://github.com/mir-group/nequip
        3). e3nn.nn.models.gate_points_2101.Convolution
"""

def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)

class Convolution(torch.nn.Module):
    r"""equivariant convolution

    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input node features

    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes

    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes

    irreps_node_output : `e3nn.o3.Irreps` or None
        representation of the output node features

    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer

    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self, irreps_node_input, irreps_node_attr, irreps_edge_attr, irreps_node_output, fc_neurons, 
        irreps_query, irreps_key, invariant_layers=3, invariant_neurons=8,
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        self.lin1 = Linear(
            irreps_in=self.irreps_node_input,
            irreps_out=self.irreps_node_input,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        assert irreps_mid.dim > 0, (
            f"irreps_node_input={self.irreps_node_input} time irreps_edge_attr={self.irreps_edge_attr} produces nothing "
            f"in irreps_node_output={self.irreps_node_output}"
        )
        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet([fc_neurons] + invariant_layers*[invariant_neurons]+[tp.weight_numel], torch.nn.functional.silu)
        self.tp = tp

        self.lin2 = Linear(
            # irreps_mid has uncoallesed irreps because of the uvu instructions,
            # but there's no reason to treat them seperately for the Linear
            # Note that normalization of o3.Linear changes if irreps are coallesed
            # (likely for the better)
            irreps_in=irreps_mid.simplify(),
            irreps_out=self.irreps_node_output,
            internal_weights=True,
            shared_weights=True,
        )
        self.h_q = Linear(self.irreps_node_input, irreps_query)
        self.tp_k = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_edge_attr, irreps_key, shared_weights=False)
        self.fc_k = FullyConnectedNet([fc_neurons] + invariant_layers*[invariant_neurons]+[self.tp_k.weight_numel], act=torch.nn.functional.silu) #MLP of basis functions
        self.dot = FullyConnectedTensorProduct(irreps_query, irreps_key, "0e")  #tensor product to rank 0 (scalar) is equivalent to dot.

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:

        node_self_connection = self.sc(node_input, node_attr) #kind of resnet
        node_features = self.lin1(node_input)
        q = self.h_q(node_features)
        k = self.tp_k(node_features[edge_src], edge_attr, self.fc_k(edge_scalars)) 
        v = self.tp(node_features[edge_src], edge_attr, self.fc(edge_scalars))
        x = self.dot(q[edge_dst], k)
        exp = torch.exp(x - torch.max(x))
        z = scatter(exp, edge_dst, dim_size=node_input.shape[0])
        z[z == 0] = 1 
        alpha = exp / z[edge_dst]
        node_features = scatter(alpha * v, edge_dst, dim_size=node_input.shape[0])

        node_conv_out = self.lin2(node_features)

        return node_self_connection + node_conv_out
    

    