# DosNet

This project is to predict the electronic density of states based on SE(3) equivalent transformer.

Github repository: [https://github.com/JiaxuanLiu-Arsko/DosNet/](https://github.com/JiaxuanLiu-Arsko/DosNet/)

Pretrained Model file and model config are uploaded to the [Release](https://github.com/JiaxuanLiu-Arsko/DosNet/releases/tag/v0.0.1-full)

## Required Packages

The DosNet model is based on **pytorch**, **pytorch_geometric**, and **e3nn**. 
Besides, 
**mp_api**, **pymatgen** are required for downloading data from Materials Project.
**ase** is required for creating molecule graph.
**monty**, **csv**, **scipy**, **sklearn** are required for preparing dataset.

The version of important packages is listed below:
Python                        3.10.0
torch                         2.2.2+cu118
torch_geometric               2.5.2
e3nn                          0.5.1
ase                           3.22.1
pymatgen                      2024.3.1
mp-api                        0.41.2

## Data requisition and Preprocess

To get training data from Materials Project, follow the steps

1. Run download_MP.py to get a summary file *mp_dos.json.gz*.

2. Run get_MP_data.py to download the raw data from Materials Project. Then files with name *mp-xxx.vasp* and *mp-xxx.npy* will be find in **structure_path** and **dos_raw** folder seperately. Simultaneously, csv file with all structure id will be found as **target_file**.

3. Run preprocess_DOS.py to get preprocessed data. Then files with name *mp-xxx_dosd.csv* will be found in **post_path** folder.


**target_file**: the file that stores the downloaded srtructure id

**dos_raw**: the folder that stores downloaded raw dos

**structure_path**: the folder that stores downloaded crystal structures

**post_path**: the folder that stores preprocessed dos data

Notice: So far only data from Material Project is supported for this process. To train on data from other resources, it is recommended to adjust the original script.

## Train the model

Training script is provided for both single-card and multi-card.

To train the model with a single-card, see script *main.py*.

To train the model with multi-card, see script *multi_card.py*.

In the script, **target_file**, **structure_path**, **post_path** are the same as paths in data download and preprocessing.

Other parameters control the structure of the network, including

**num_types**: types of atoms in dataset

**irreps_list**: irreps order and number of channels in each layer

**r_max**: radius cut for creating molecule graph

**fc_neurons**: number of neurons per layer in the fully connected network for atom embedding

**num_basis**: dos vector dimension

Notice: 
when trained on a new dataset, it is required to adjust n = 118 based on the dataset. 

## Utilize trained model

To utilize the trained model for dos of a structure, prepare POSCAR file and run

python _from_poscar_to_dos.py POSCAR xxx.pth xxx.yaml xxx

POSCAR: POSCAR file

xxx.pth: model file

xxx.yaml: model config file

xxx: output folder for dos.txt and dos.pdf (if this folder is not found, two files will be created under the same folder as POSCAR )
