from mp_api.client import MPRester
import ase
from ase.io import vasp
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from monty.serialization import loadfn 
import os

'''
mp_doc.json.gz path:
output vasp path:
output npy path:

'''

save_path = "/work/jxliu/e3nn/DOS_net/data/" # data folder

strucure_path = "/work/jxliu/e3nn/DOS_net/data/structure/" # sub folder of data folder
dos_raw_path = "/work/jxliu/e3nn/DOS_net/data/dos_raw/" # sub folder of data folder
target_path = "/work/jxliu/e3nn/DOS_net/data/" # data folder

'''
To be add: output to track download procress
how many data points checked, how many saved
'''



API_KEY='XXX'
mpr = MPRester('WSixK0Or5Ro47ND37DnNcd49xanMqUsV')

#print('start')

###get API key from materials project login dashboard online

data = loadfn(os.path.join(save_path, f"mp_doc.json.gz"))
print(len(data), data[0])
adaptor = AseAtomsAdaptor()
lengths=[]

count = 0
loop = 0

#for z in range(0, len(data)):
for z in range(300, len(data)):
    energy=data[z]["formation_energy_per_atom"]
    #print(energy)
    if energy == None:
      loop = loop + 1
      continue
    if energy > -500 and energy < 100:        
        try:
          out_temp=mpr.get_dos_by_material_id(data[z]["material_id"])
        except Exception as e:
          print(e) 
          loop = loop+1
          continue    
               
        if out_temp != None:
          dos_temp=out_temp.get_site_spd_dos(out_temp.structure[0])    
          orb = list(dos_temp.keys())
          fermi = dos_temp[orb[0]].efermi
          length=dos_temp[list(dos_temp.keys())[0]].get_densities(spin=None).shape
          dos=np.zeros((len(out_temp.structure), 4, length[0]))   
          for i in range(0, len(out_temp.structure)):  
            dos_temp=out_temp.get_site_spd_dos(out_temp.structure[i])
            shape=dos_temp[orb[0]].get_densities(spin=None).shape
            ##sum over orbitals
            for j in range(0, 4):
              if j == 0:
                dos[i,j,:]=dos_temp[orb[j]].energies - fermi
              else:
                dos[i,j,:]=dos_temp[orb[j-1]].get_densities(spin=None)    
          
          ##write structure and dos                
          ase_structure = adaptor.get_atoms(out_temp.structure)      
          if length[0] == 2001:   
            np.save(dos_raw_path +str(data[z]["material_id"])+'.npy', dos)            
            ase.io.vasp.write_vasp(strucure_path + str(data[z]["material_id"])+".vasp", ase_structure, vasp5=True)        
            ##placeholder  
            with open(target_path + 'targets.csv', 'a') as f:
              f.write(str(data[z]["material_id"])+','+str(energy)+','+str('0.00') + '\n')  
    
    count = count + 1
    loop = loop+1
    if loop%1000 == 0:
      print(loop , " checked ", count, " saved")

'''
92000 checked， 5w+ saved
'''