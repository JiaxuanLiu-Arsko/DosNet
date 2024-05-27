from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import numpy as np
import os
import torch
import csv
import ase
from ase import io
#from process import get_dos_features


target_file = "/work/jxliu/e3nn/DOS_net/DosNet/data/targets.csv"
dos_raw_path = "/work/jxliu/e3nn/DOS_net/DosNet/data/dos_raw"
structure_path = "/work/jxliu/e3nn/DOS_net/DosNet/data/structure/"
post_path = "/work/jxliu/e3nn/DOS_net/DosNet/data/preprocessed"
#f_run = "/Users/hm-t03-mac2/Documents/py_proj/band_fit/dataset/data_prepocessed/trace_run.csv"

def get_dos_features(x, dos):
    dos=torch.abs(dos) 
    center=torch.sum(x*dos, axis=1)/torch.sum(dos, axis=1)
    x_offset = torch.repeat_interleave(x[np.newaxis,:], dos.shape[0], axis=0)-center[:,None]
    width = torch.diagonal(torch.mm((x_offset**2), dos.T))/torch.sum(dos, axis=1)
    skew = torch.diagonal(torch.mm((x_offset**3), dos.T))/torch.sum(dos, axis=1)/width**(1.5)
    kurtosis = torch.diagonal(torch.mm((x_offset**4), dos.T))/torch.sum(dos, axis=1)/width**(2)
    
    #find zero index (fermi leve)
    zero_index = torch.abs(x-0).argmin().long()
    ef_states = torch.sum(dos[:,zero_index-20:zero_index+20], axis=1)*abs(x[0]-x[1])
    return torch.stack((center, width, skew, kurtosis, ef_states), axis=1)

with open(target_file) as f:
    reader = csv.reader(f)
    target_data = [row for row in reader]

data_format = "vasp"
print(len(target_data), " structures in total")

for index in range(len(target_data)):
    #if(index%100==0):
    #    f_ = open(f_run, "w")
    #    f_.write(str(index))

    structure_id = target_data[index][0]

    if data_format != "db":
        ase_crystal = ase.io.read(
            os.path.join(
                structure_path, structure_id + "." + data_format
            )
        )
    #print(ase_crystal)
    #print(len(ase_crystal))
    dos_read = np.load(os.path.join(dos_raw_path, structure_id + ".npy")) 
    #print(dos_read.shape)
    #print(dos_read)
    dos_sum = np.sum(dos_read[:,1:,:], axis=1)
    #print(dos_sum.shape)
    #print(dos_sum)
    #print(np.concatenate((dos_read[0,0,:][np.newaxis,:], dos_sum), axis=0))
    #print(np.concatenate((dos_read[0,0,:][np.newaxis,:], dos_sum), axis=0).shape)
    ##np.savetxt((os.path.join(post_path, structure_id + "_processed_0.csv")), np.concatenate((dos_read[0,0,:][np.newaxis,:], dos_sum), axis=0), delimiter=",") 
    for i in range(0, len(ase_crystal)):
        dos_sum[i,:] = gaussian_filter1d(dos_sum[i,:], sigma=7) 
    #interpolation and shift energy window     
    dos_length=400
    dos=np.zeros((len(ase_crystal),dos_length))
    for i in range(0, len(ase_crystal)):
        xfit=dos_read[i,0,:]
        yfit=dos_sum[i,:]     
        dos_fit = interpolate.interp1d(xfit, yfit, kind="linear", bounds_error=False, fill_value=0)
        xnew = np.linspace(-10, 10, dos_length) 
        dos[i,:]=dos_fit(xnew)
        #print(dos[i,:]) 
    #print(dos[0].shape)
    #print(dos.shape) 
    y = torch.Tensor(dos)
    #print(dos[0,:])
    #print(dos.shape)
    ##np.savetxt((os.path.join(data_path, structure_id + "_processed_1.csv")), dos, delimiter=",") 
    
    dos_features = get_dos_features(torch.Tensor(xnew), y)
    #print(dos_features.shape)       
    scaling = np.max(dos, axis=1)      
    for i in range(0, len(ase_crystal)):
        dos[i,:] = dos[i,:]/scaling[i] 
    np.savetxt((os.path.join(post_path, structure_id + "_dosd.csv")), dos, delimiter=",")
    scaling = torch.tensor(scaling)
    feature_all = torch.cat((scaling.unsqueeze(1), dos_features), axis=1).numpy()
    np.savetxt((os.path.join(post_path, structure_id + "_feature.csv")), feature_all, delimiter=",")

    #print(feature_all)
    #if 0 in scaling:
    #    print(structure_id)
    #if np.isnan(np.min(dos)):
    #    print(structure_id)

    #print(dos[0,:])
    #print(dos.shape)
    #print(scaling)
    #print(scaling.shape)
    #np.savetxt((os.path.join(data_path, structure_id + "_processed_final.csv")), dos, delimiter=",")        
    #data.dos_scaled=torch.Tensor(dos)       
    #data.scaling_factor = torch.Tensor(scaling)




