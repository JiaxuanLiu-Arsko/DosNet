import torch
from ase.io.vasp import read_vasp
from matplotlib import pyplot as plt
import argparse
import os
from model._dos_net import DosNet
from torch_geometric.data import Data
from data._dataset import GraphData
from torch.nn.functional import one_hot
import numpy as np
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--poscar', help='path to POSCAR file')
parser.add_argument('--model', help='path to model file')
parser.add_argument('--config', help='path to model config')
parser.add_argument('--output', help='path to output file')
args = parser.parse_args()

# check parser
assert os.path.exists(args.poscar)
assert os.path.exists(args.model)

n = 118
n_basis = 400
layer_n_list = [n, int((n+n_basis)/2), n_basis]
layer_l_list = [3, 3, 0]
qk_irreps = f"{n}x0e+{n}x1o+{n}x2e"
layer_input = f"{n}x0e"
irreps_list = []

config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

# load model
model = DosNet(config)
if torch.cuda.is_available():
    model.cuda()
model.load_state_dict(torch.load(args.model))
model.eval()

# load poscar
r_cut = 6

g = GraphData(r_cut, args.poscar)
x=torch.tensor(one_hot(torch.tensor(g.symbols), num_classes=118),dtype=torch.float32)
edge_index=torch.stack([torch.tensor(g.edge_src, dtype=torch.long), torch.tensor(g.edge_dst, dtype=torch.long)], dim=0)
edge_shift = g.edge_shift
edge_vec = g.edge_vec
lattice = g.lattice
pos = g.positions
data = Data(x=x.clone(), node_attr=x ,edge_index=edge_index, edge_vec=edge_vec, edge_shift=edge_shift, lattice=lattice, pos=pos)
if torch.cuda.is_available():
    data.cuda()

# get predicted dos
dos = torch.mean(model(data), dim=0)
if torch.cuda.is_available():
    dos = dos.cpu()
dos = dos.detach().numpy()
# save dos
x = np.linspace(-10, 10, 400)
plt.plot(x, dos)
if not os.path.exists(args.output):
    args.output = os.path.splitext(args.poscar)[0] + '.dos'
    plt.savefig(os.path.splitext(args.poscar)[0] + ".pdf")
    print("output path not found, save dos to ", args.output)
else:
    np.savetxt(os.path.join(args.output, "dos.txt"), dos, delimiter=",")
    plt.savefig(os.path.join(args.output, "dos.pdf"))

'''
    /Users/hm-t03-mac2/Documents/py_proj/DosNet_pl/script/Ac3Tl.poscar
    /Users/hm-t03-mac2/Documents/py_proj/DosNet_pl/model/model_36.pth
    xxxx
'''