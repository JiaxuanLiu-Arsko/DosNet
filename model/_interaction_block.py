import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, SphericalHarmonics
from ._points_convolution import Convolution
from ._bessel_basis import BesselBasis
from ._cutoffs import PolynomialCutoff
from torch_geometric.data import Data

class InteractionLayer(torch.nn.Module):
    '''
    irreps_list: input, edge, output
    irreps["edge"] should be list of int
    '''
    def __init__(self, irreps_list:list, irreps_node_attr, r_max, fc_neurons = 8, invariant_layers=3, invariant_neurons=8) -> None:
        super().__init__()
        self.bessel = BesselBasis(r_max, num_basis=8, trainable=True)
        self.cutoff = PolynomialCutoff(r_max=r_max)
        self.conv_layers = torch.nn.ModuleList([])
        self.harmonics = torch.nn.ModuleList([])
        for irreps in irreps_list:
            self.conv_layers.append(Convolution(irreps["input"], irreps_node_attr, irreps["edge"], irreps["output"], fc_neurons, 
                                                irreps["query"], irreps["key"], invariant_layers, invariant_neurons))
            self.harmonics.append(SphericalHarmonics(irreps["edge"], True, "component", irreps_in="1o"))
        # notes about spherical harmonics
        # l (int or list of int) – degree of the spherical harmonics.
        # x (torch.Tensor) – tensor of shape (..., 3).
        # normalize (bool) – whether to normalize the x to unit vectors
        # normalization ({'integral', 'component', 'norm'})
        
    def forward(self, node_feature, node_attr, edge_src, edge_dst, edge_vec):
        edge_norm = torch.linalg.norm(edge_vec, dim=-1)
        edge_scalars = self.bessel(edge_norm)*self.cutoff(edge_norm)[:, None]
        for i, conv_layer in enumerate(self.conv_layers):
            edge_attr = self.harmonics[i](edge_vec) # sperical harmonics rep of edge vector
            node_feature = conv_layer(node_feature, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)
        return node_feature
        