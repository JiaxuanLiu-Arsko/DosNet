#test_dataset

import ase
from ase.io.vasp import read_vasp
from ase import neighborlist as neigh
from e3nn.math import complete_basis
import torch
import numpy as np
from torch.utils.data import random_split
from torch.nn.functional import one_hot
import os
from numpy import genfromtxt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import csv

class GraphData():
    '''
    Same as the GraphData in the original code. Read the structure from vasp file and convert is into graph data.

    Input: r_cut, fname
    
    Output: GraphData object of the molecule structure

    # the following is not strict tensor dim but to illustrate what is contained in GraphData
    positions: (N_atom, 3)
    symbols: (N_taom,)
    edge_src: (edge_num, )
    edge_dst: (edge_num, )
    edge_shift
    edge_vec: (edge_num, 3)
    lattice: (1, 3, 3)
    '''
    def __init__(self, r_cut, fname=None) -> None:
        self.r_cut = r_cut
        if fname:
            self.load_from_vasp(fname)

    def load_from_atoms(self, atoms:ase.Atoms):
        edge_src, edge_dst, edge_shift, edge_vec = neigh.neighbor_list("ijSD", atoms, cutoff=self.r_cut, self_interaction=False)
        self.positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        self.symbols = atoms.get_chemical_symbols()
        self.edge_src = edge_src
        self.edge_dst = edge_dst
        self.edge_shift = torch.tensor(edge_shift, dtype=torch.float32)
        self.edge_vec = torch.tensor(edge_vec, dtype=torch.float32)
        self.lattice = torch.tensor(atoms.get_cell().array, dtype=torch.float32).unsqueeze(0)

    def load_from_vasp(self, fname):
        self.load_from_atoms(read_vasp(fname))


class DosData():
    '''
    Read in DosData from the dos file and dos feature file.

    Input: preprocessed dos path, id

    Output: DosData object of the molecule structure
    ---
    dos: (atom_num, 400)
    scale: (atom_num, )
    feature:(atom_num, 5)
    ---
    '''
    def __init__(self, path, id) -> None:
        dos_file = os.path.join(path, f"{id}_dosd.csv")
        scale_feature_file = os.path.join(path, f"{id}_feature.csv")
        assert os.path.exists(dos_file)
        assert os.path.exists(scale_feature_file)
        self.dos_vec, self.scale, self.feature = self.load_data(dos_file, scale_feature_file)
        #print(self.dos_vec.shape)
        #print(self.scale.shape)
        #print(self.feature.shape)
    
    def load_data(self, dos_file, scale_feature_file):
        dos_vec = genfromtxt(dos_file, delimiter=',')
        scale_feature = genfromtxt(scale_feature_file, delimiter=',')
        if len(dos_vec.shape) == 1:
            dos_vec = dos_vec[np.newaxis, :]
            scale_feature = scale_feature[np.newaxis, :]

        return torch.tensor(dos_vec, dtype=torch.float32), torch.tensor(scale_feature[:,0], dtype=torch.float32),\
            torch.tensor(scale_feature[:,1:6], dtype=torch.float32)

class Dataset():
    '''
    load in Dataset from the id file, structure path and dos path
    ---
    dos: list of DosData objects
    graph: list of GraohData objects
    material: list of structure id
    n_data: number of structures loaded
    atom_type: all atom types in the dataset
    n_type: number of atom types in the dataset
    r_cut: radius cut for creating the graph
    ---
    '''
    def __init__(self, r_cut=5) -> None:
        '''
        Initialize the dataset by radius cut
        '''
        self.dos = []
        self.graph = []

        self.n_data = 0
        self.atom_type = {}
        self.n_type = 0
        self.r_cut = r_cut
        self.material = []

    def load_from_dir(self, id_file, structure_path, dos_path):
        '''
        Input: id_file, structure_path, dos_path
        Output: load in molecule structures
        '''
        with open(id_file) as f:
            reader = csv.reader(f)
            StructureID = [row for row in reader]
        for index in range(len(StructureID)):
            structure_id = StructureID[index][0]
            #print(structure_id) 
            self.material.append(structure_id)
            self.graph.append(GraphData(r_cut=self.r_cut, fname=os.path.join(structure_path, f"{structure_id}.vasp")))
            self.dos.append(DosData(dos_path, structure_id))
            self.n_data += 1
            for symbol in self.graph[-1].symbols :
                if symbol not in self.atom_type.keys():
                    self.atom_type[symbol] = self.n_type
                    self.n_type += 1

    def __len__(self):
        return len(self.dos)

    def __getitem__(self, idx):
        crystal_structure = self.graph[idx]
        dos = self.dos[idx]
        return crystal_structure, dos

    def get_data_Loader(self, batch_size:int):
        '''
        Get train_loader and valid_loader
        '''
        dataset = []
        for i, g in enumerate(self.graph):
            #print(g.symbols)
            x=torch.tensor(one_hot(torch.tensor([self.atom_type[symbol] for symbol in g.symbols]), num_classes=len(self.atom_type.keys())),dtype=torch.float32)
            edge_index=torch.stack([torch.tensor(g.edge_src, dtype=torch.long), torch.tensor(g.edge_dst, dtype=torch.long)], dim=0)
            edge_shift = g.edge_shift
            edge_vec = g.edge_vec
            lattice = g.lattice
            pos = g.positions
            dos_vec = self.dos[i].dos_vec
            scale = self.dos[i].scale
            feature = self.dos[i].feature
            # according to geoData class, here only the first two are variables in Data, else are **kwards
            data = Data(x=x.clone(), node_attr=x ,edge_index=edge_index, edge_vec=edge_vec, edge_shift=edge_shift, lattice=lattice,\
                    dos_vec = dos_vec, scale = scale, feature = feature, pos=pos, mat=self.material[i])
            #print(data)
            dataset.append(data)
        #print(len(dataset))

        train_size = int(0.7 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device = 'cpu'))
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device = 'cpu'))
        return train_dataloader, valid_dataloader

'''
target_file = "/Users/hm-t03-mac2/Documents/py_proj/DosNet/data/targets.csv"
structure_path = "/Users/hm-t03-mac2/Documents/py_proj/DosNet/data/structure/"
post_path = "/Users/hm-t03-mac2/Documents/py_proj/DosNet/data/preprocessed/"
dataset = Dataset()
dataset.load_from_dir(target_file , structure_path, post_path)
train_dataloader, valid_dataloader = dataset.get_data_Loader(10)
for i, data in enumerate(train_dataloader):
    print(data)
    if i == 1:
        break
'''
