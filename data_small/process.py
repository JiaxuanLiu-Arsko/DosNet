import numpy as np
import torch


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