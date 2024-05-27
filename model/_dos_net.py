import torch
from e3nn.o3 import Linear
from e3nn.nn import FullyConnectedNet
from torch_geometric.data import Data, Batch
from ._interaction_block import InteractionLayer
from torch_geometric.nn.pool import global_mean_pool


class DosNet(torch.nn.Module):
    def __init__(self, config) -> None:
        '''
        config list include:
        每层的表示种类和channel数目
        基函数选择
        weight的全联接网络的层数和每层的node数目
        '''
        super().__init__()
        self.element_embbeding = FullyConnectedNet(3*[config["num_types"]], torch.nn.functional.silu)
        self.inter_layer = InteractionLayer(irreps_list=config["irreps_list"], irreps_node_attr=f"{config['num_types']}x0e", 
                                            r_max=config["r_max"], fc_neurons=config["fc_neurons"])
        self.output_block = Linear(config["irreps_list"][-1]["output"], config["irreps_list"][-1]["output"])
    
    def forward(self, data:Batch|Data):
        '''
        what about not using data. ?
        '''
        data.x = self.element_embbeding(data.x)
        data.node_attr = data.x.clone()
        data.x = self.inter_layer(data.x, data.node_attr, data.edge_index[1], data.edge_index[0], data.edge_vec)
        output = torch.nn.functional.silu(self.output_block(data.x))
        return output
    #  output: dos vect (N_atoms, 400)
